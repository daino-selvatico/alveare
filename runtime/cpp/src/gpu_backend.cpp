#include "alveare/gpu_backend.h"
#include "alveare/cpu_backend.h"
#include "alveare/spv_shaders.h"

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <algorithm>

namespace alveare {

namespace {

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    // Fallback: any matching type
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (type_filter & (1 << i)) {
            return i;
        }
    }
    return 0;
}

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t size = 0;

    void destroy(VkDevice device) {
        if (mapped) {
            vkUnmapMemory(device, memory);
            mapped = nullptr;
        }
        if (buffer) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
        size = 0;
    }
};

GpuBuffer create_buffer(VkDevice device, VkPhysicalDevice physical_device, size_t size, VkBufferUsageFlags usage) {
    GpuBuffer buf;
    buf.size = size;

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size;
    buf_info.usage = usage;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &buf_info, nullptr, &buf.buffer) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateBuffer failed");
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buf.buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        physical_device,
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    if (vkAllocateMemory(device, &alloc_info, nullptr, &buf.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        throw std::runtime_error("vkAllocateMemory failed");
    }

    vkBindBufferMemory(device, buf.buffer, buf.memory, 0);
    vkMapMemory(device, buf.memory, 0, size, 0, &buf.mapped);
    return buf;
}

} // anonymous namespace

struct GpuWeight {
    int N = 0;
    int K = 0;
    GpuBuffer buffer;
    std::vector<uint8_t> host_copy; // Keep host copy for batched/fallback operations
};

struct GpuBackend::Impl {
    bool vulkan_initialized = false;
    std::string device_name = "CPU Fallback (Vulkan Init Failed)";
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    uint32_t compute_queue_family = 0;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    VkShaderModule gemv_shader = VK_NULL_HANDLE;
    VkPipeline gemv_pipeline = VK_NULL_HANDLE;

    VkShaderModule gemm_shader = VK_NULL_HANDLE;
    VkPipeline gemm_pipeline = VK_NULL_HANDLE;

    GpuBuffer input_buffer;
    GpuBuffer output_buffer;

    std::vector<GpuWeight> weights;
    CpuBackend cpu_fallback;
    std::mutex mutex;

    Impl() {
        try {
            init_vulkan();
        } catch (const std::exception& e) {
            std::cerr << "[GpuBackend] Warning: Vulkan initialization failed (" << e.what()
                      << "). Using CPU fallback mode.\n";
            vulkan_initialized = false;
        }
    }

    ~Impl() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            input_buffer.destroy(device);
            output_buffer.destroy(device);
            for (auto& w : weights) {
                w.buffer.destroy(device);
            }
            if (gemv_pipeline) vkDestroyPipeline(device, gemv_pipeline, nullptr);
            if (gemm_pipeline) vkDestroyPipeline(device, gemm_pipeline, nullptr);
            if (gemv_shader) vkDestroyShaderModule(device, gemv_shader, nullptr);
            if (gemm_shader) vkDestroyShaderModule(device, gemm_shader, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, nullptr);
            if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    void init_vulkan() {
        // 1. Create Vulkan Instance
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "alveare_runtime";
        app_info.applicationVersion = VK_MAKE_VERSION(3, 0, 0);
        app_info.pEngineName = "alveare_gpu_engine";
        app_info.engineVersion = VK_MAKE_VERSION(3, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo inst_info{};
        inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        inst_info.pApplicationInfo = &app_info;

        if (vkCreateInstance(&inst_info, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateInstance failed");
        }

        // 2. Pick Physical Device (prefer Radeon 890M / integrated / discrete GPU)
        uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
        if (dev_count == 0) throw std::runtime_error("No Vulkan physical devices found");

        std::vector<VkPhysicalDevice> devices(dev_count);
        vkEnumeratePhysicalDevices(instance, &dev_count, devices.data());

        for (auto pd : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            // Ignore CPU llvmpipe if an integrated or discrete GPU is present
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device = pd;
                device_name = std::string(props.deviceName) + " (Vulkan 1.4 Native)";
                break;
            }
        }
        if (!physical_device && !devices.empty()) {
            physical_device = devices[0];
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_device, &props);
            device_name = std::string(props.deviceName) + " (Vulkan)";
        }

        // 3. Find Compute Queue Family
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

        bool found_queue = false;
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_family = i;
                found_queue = true;
                break;
            }
        }
        if (!found_queue) throw std::runtime_error("No compute queue family found");

        // 4. Create Logical Device
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = compute_queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        VkDeviceCreateInfo dev_info{};
        dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dev_info.queueCreateInfoCount = 1;
        dev_info.pQueueCreateInfos = &queue_info;

        if (vkCreateDevice(physical_device, &dev_info, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateDevice failed");
        }
        vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);

        // 5. Command Pool & Buffer & Fence
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = compute_queue_family;
        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateCommandPool failed");
        }

        VkCommandBufferAllocateInfo cmd_info{};
        cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_info.commandPool = command_pool;
        cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_info.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cmd_info, &command_buffer);

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device, &fence_info, nullptr, &fence);

        // 6. Descriptor Set Layout (3 SSBOs: weights, input, output)
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 3;
        layout_info.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &desc_layout) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateDescriptorSetLayout failed");
        }

        // Pipeline Layout with push constants (2 ints for GEMV: N, K; or 3 ints for GEMM: B, N, K)
        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = sizeof(int) * 4;

        VkPipelineLayoutCreateInfo pl_info{};
        pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_info.setLayoutCount = 1;
        pl_info.pSetLayouts = &desc_layout;
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;
        if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            throw std::runtime_error("vkCreatePipelineLayout failed");
        }

        // 7. Descriptor Pool & Set
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 32;

        VkDescriptorPoolCreateInfo dp_info{};
        dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp_info.flags = 0;
        dp_info.maxSets = 8;
        dp_info.poolSizeCount = 1;
        dp_info.pPoolSizes = &pool_size;
        vkCreateDescriptorPool(device, &dp_info, nullptr, &desc_pool);

        VkDescriptorSetAllocateInfo ds_info{};
        ds_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_info.descriptorPool = desc_pool;
        ds_info.descriptorSetCount = 1;
        ds_info.pSetLayouts = &desc_layout;
        vkAllocateDescriptorSets(device, &ds_info, &desc_set);

        // 8. Create Compute Shader Modules & Pipelines
        VkShaderModuleCreateInfo sm_info{};
        sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_info.codeSize = gpu_shaders::gemv_q4_0_spv_size;
        sm_info.pCode = gpu_shaders::gemv_q4_0_spv;
        if (vkCreateShaderModule(device, &sm_info, nullptr, &gemv_shader) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateShaderModule (GEMV) failed");
        }

        VkComputePipelineCreateInfo cp_info{};
        cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_info.layout = pipeline_layout;
        cp_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp_info.stage.module = gemv_shader;
        cp_info.stage.pName = "main";
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &gemv_pipeline) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateComputePipelines (GEMV) failed");
        }

        // GEMM pipeline
        sm_info.codeSize = gpu_shaders::gemm_q4_0_spv_size;
        sm_info.pCode = gpu_shaders::gemm_q4_0_spv;
        if (vkCreateShaderModule(device, &sm_info, nullptr, &gemm_shader) == VK_SUCCESS) {
            cp_info.stage.module = gemm_shader;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &gemm_pipeline);
        }

        // 9. Allocate persistent input/output activation buffers (large enough for 64K tokens)
        input_buffer = create_buffer(device, physical_device, 64 * 1024 * 1024,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        output_buffer = create_buffer(device, physical_device, 64 * 1024 * 1024,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        vulkan_initialized = true;
        std::cout << "[GpuBackend] Successfully initialized " << device_name << "!\n";
    }
};

GpuBackend::GpuBackend() : impl_(std::make_unique<Impl>()) {}
GpuBackend::~GpuBackend() = default;

std::string GpuBackend::name() const {
    return impl_->device_name;
}

bool GpuBackend::is_vulkan_active() const {
    return impl_->vulkan_initialized;
}

WeightHandle GpuBackend::create_gemv_weight(int N, int K, const void* packed, size_t nbytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    WeightHandle handle = static_cast<WeightHandle>(impl_->weights.size());
    GpuWeight gw;
    gw.N = N;
    gw.K = K;
    const uint8_t* p = static_cast<const uint8_t*>(packed);
    gw.host_copy.assign(p, p + nbytes);

    if (impl_->vulkan_initialized) {
        try {
            gw.buffer = create_buffer(impl_->device, impl_->physical_device, nbytes,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            std::memcpy(gw.buffer.mapped, packed, nbytes);
        } catch (const std::exception& e) {
            std::cerr << "[GpuBackend] Buffer creation failed: " << e.what() << "\n";
        }
    }
    impl_->weights.push_back(std::move(gw));
    return handle;
}

WeightHandle GpuBackend::create_ffn_fused_weight(int H, int I, const std::string& activation,
                                                 const void* packed, size_t nbytes) {
    // Registered in host fallback/internal structure
    return impl_->cpu_fallback.create_ffn_fused_weight(H, I, activation, packed, nbytes);
}

void GpuBackend::run_gemv(int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) {
    auto t0 = std::chrono::steady_clock::now();
    if (!impl_->vulkan_initialized || w >= impl_->weights.size() || !impl_->weights[w].buffer.buffer) {
        // Fallback to optimized CPU AVX2 if Vulkan not ready for this shape
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto& gw = impl_->weights[w];

    // 1. Copy activation to GPU mapped memory (K bf16 elements = K * 2 bytes)
    size_t in_bytes = static_cast<size_t>(K) * sizeof(bf16);
    std::memcpy(impl_->input_buffer.mapped, x_bf16, in_bytes);

    // 2. Update Descriptors: 0: weight, 1: input, 2: output
    VkDescriptorBufferInfo buf_infos[3]{};
    buf_infos[0].buffer = gw.buffer.buffer;
    buf_infos[0].offset = 0;
    buf_infos[0].range = gw.buffer.size;

    buf_infos[1].buffer = impl_->input_buffer.buffer;
    buf_infos[1].offset = 0;
    buf_infos[1].range = in_bytes;

    size_t out_bytes = static_cast<size_t>(N) * sizeof(bf16);
    buf_infos[2].buffer = impl_->output_buffer.buffer;
    buf_infos[2].offset = 0;
    buf_infos[2].range = out_bytes;

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = impl_->desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(impl_->device, 3, writes, 0, nullptr);

    // 3. Record & Submit Dispatch
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(impl_->command_buffer, &begin_info);

    vkCmdBindPipeline(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->gemv_pipeline);
    vkCmdBindDescriptorSets(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            impl_->pipeline_layout, 0, 1, &impl_->desc_set, 0, nullptr);

    int pc_data[2] = { N, K };
    vkCmdPushConstants(impl_->command_buffer, impl_->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_data), pc_data);

    // Dispatch N workgroups (1 per row)
    vkCmdDispatch(impl_->command_buffer, static_cast<uint32_t>(N), 1, 1);
    vkEndCommandBuffer(impl_->command_buffer);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &impl_->command_buffer;

    vkResetFences(impl_->device, 1, &impl_->fence);
    vkQueueSubmit(impl_->compute_queue, 1, &submit_info, impl_->fence);
    vkWaitForFences(impl_->device, 1, &impl_->fence, VK_TRUE, UINT64_MAX);

    // 4. Read output back from mapped memory
    std::memcpy(y_bf16, impl_->output_buffer.mapped, out_bytes);

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    elapsed_seconds_ += dt;
    ++total_calls_;
}

void GpuBackend::run_gemv_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                const void* x_bf16, void* y_bf16_concat) {
    auto t0 = std::chrono::steady_clock::now();
    bf16* out_ptr = static_cast<bf16*>(y_bf16_concat);
    for (size_t i = 0; i < weights.size(); ++i) {
        run_gemv(N, K, weights[i], x_bf16, out_ptr + i * N);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    elapsed_seconds_ += dt;
}

void GpuBackend::run_gemv_multi_in_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                         const std::vector<const void*>& x_ptrs, void* y_bf16_concat) {
    auto t0 = std::chrono::steady_clock::now();
    bf16* out_ptr = static_cast<bf16*>(y_bf16_concat);
    for (size_t i = 0; i < weights.size(); ++i) {
        const void* x_in = (i < x_ptrs.size()) ? x_ptrs[i] : x_ptrs[0];
        run_gemv(N, K, weights[i], x_in, out_ptr + i * N);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    elapsed_seconds_ += dt;
}

void GpuBackend::run_gemm(int B, int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) {
    auto t0 = std::chrono::steady_clock::now();
    if (!impl_->vulkan_initialized || w >= impl_->weights.size() || !impl_->weights[w].buffer.buffer || !impl_->gemm_pipeline) {
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto& gw = impl_->weights[w];

    size_t in_bytes = static_cast<size_t>(B) * K * sizeof(bf16);
    std::memcpy(impl_->input_buffer.mapped, x_bf16, in_bytes);

    VkDescriptorBufferInfo buf_infos[3]{};
    buf_infos[0].buffer = gw.buffer.buffer;
    buf_infos[0].offset = 0;
    buf_infos[0].range = gw.buffer.size;

    buf_infos[1].buffer = impl_->input_buffer.buffer;
    buf_infos[1].offset = 0;
    buf_infos[1].range = in_bytes;

    size_t out_bytes = static_cast<size_t>(B) * N * sizeof(bf16);
    buf_infos[2].buffer = impl_->output_buffer.buffer;
    buf_infos[2].offset = 0;
    buf_infos[2].range = out_bytes;

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = impl_->desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(impl_->device, 3, writes, 0, nullptr);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(impl_->command_buffer, &begin_info);

    vkCmdBindPipeline(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->gemm_pipeline);
    vkCmdBindDescriptorSets(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            impl_->pipeline_layout, 0, 1, &impl_->desc_set, 0, nullptr);

    int pc_data[3] = { B, N, K };
    vkCmdPushConstants(impl_->command_buffer, impl_->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_data), pc_data);

    // Dispatch (N, B, 1) workgroups
    vkCmdDispatch(impl_->command_buffer, static_cast<uint32_t>(N), static_cast<uint32_t>(B), 1);
    vkEndCommandBuffer(impl_->command_buffer);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &impl_->command_buffer;

    vkResetFences(impl_->device, 1, &impl_->fence);
    vkQueueSubmit(impl_->compute_queue, 1, &submit_info, impl_->fence);
    vkWaitForFences(impl_->device, 1, &impl_->fence, VK_TRUE, UINT64_MAX);

    std::memcpy(y_bf16, impl_->output_buffer.mapped, out_bytes);

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    elapsed_seconds_ += dt;
    ++total_calls_;
}

void GpuBackend::run_gemm_streamed(int B, int N, int K, const void* packed, size_t nbytes,
                                   const void* x_bf16, void* y_bf16) {
    // Use multi-threaded CPU path for transient host-streamed weights
    impl_->cpu_fallback.run_gemm_streamed(B, N, K, packed, nbytes, x_bf16, y_bf16);
}

void GpuBackend::run_ffn_fused(int H, int I, const std::string& activation, WeightHandle w,
                               const void* x_bf16, void* y_bf16) {
    auto t0 = std::chrono::steady_clock::now();
    impl_->cpu_fallback.run_ffn_fused(H, I, activation, w, x_bf16, y_bf16);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ffn_seconds_ += dt;
    elapsed_seconds_ += dt;
    ++total_calls_;
}

} // namespace alveare
