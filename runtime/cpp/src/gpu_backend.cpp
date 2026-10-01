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
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
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

    VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &buf.buffer);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkCreateBuffer failed with code " + std::to_string(res));
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

    res = vkAllocateMemory(device, &alloc_info, nullptr, &buf.memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        throw std::runtime_error("vkAllocateMemory failed with code " + std::to_string(res));
    }

    res = vkBindBufferMemory(device, buf.buffer, buf.memory, 0);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device, buf.buffer, nullptr);
        vkFreeMemory(device, buf.memory, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        buf.memory = VK_NULL_HANDLE;
        throw std::runtime_error("vkBindBufferMemory failed with code " + std::to_string(res));
    }

    res = vkMapMemory(device, buf.memory, 0, size, 0, &buf.mapped);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device, buf.buffer, nullptr);
        vkFreeMemory(device, buf.memory, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        buf.memory = VK_NULL_HANDLE;
        buf.mapped = nullptr;
        throw std::runtime_error("vkMapMemory failed with code " + std::to_string(res));
    }
    return buf;
}

} // anonymous namespace

struct GpuWeight {
    int N = 0;
    int K = 0;
    GpuBuffer buffer;
    std::vector<uint8_t> host_copy;
};

struct GpuBackend::Impl {
    bool vulkan_initialized = false;
    std::string device_name = "CPU Fallback (Vulkan Init Failed)";
    std::string init_error;
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
    VkPipeline gemv_pipeline_wave32 = VK_NULL_HANDLE;

    VkShaderModule gemm_shader = VK_NULL_HANDLE;
    VkPipeline gemm_pipeline = VK_NULL_HANDLE;
    VkPipeline gemm_pipeline_wave32 = VK_NULL_HANDLE;

    GpuBuffer input_buffer;
    GpuBuffer output_buffer;

    std::vector<GpuWeight> weights;
    CpuBackend cpu_fallback;
    mutable std::mutex mutex;

    bool supports_subgroup_control = false;
    uint32_t default_subgroup_sz = 64;
    bool force_wave32 = false;

    GpuStats stats;
    std::string last_logged_fallback_reason;

    void record_fault_fallback(const std::string& reason) {
        ++stats.fault_fallback_calls;
        stats.last_fallback_reason = reason;
        if (reason != last_logged_fallback_reason) {
            std::cerr << "[GpuBackend] Fallback to CPU: " << reason << "\n";
            last_logged_fallback_reason = reason;
        }
    }

    void ensure_buffer_capacity(GpuBuffer& buf, size_t required_bytes, VkBufferUsageFlags usage) {
        if (buf.size >= required_bytes && buf.buffer != VK_NULL_HANDLE && buf.mapped != nullptr) {
            return;
        }
        size_t new_size = std::max(required_bytes, buf.size * 2);
        if (new_size < 4 * 1024 * 1024) new_size = 4 * 1024 * 1024;
        buf.destroy(device);
        buf = create_buffer(device, physical_device, new_size, usage);
    }

    void cleanup_vulkan() noexcept {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            input_buffer.destroy(device);
            output_buffer.destroy(device);
            for (auto& w : weights) {
                w.buffer.destroy(device);
            }
            if (gemv_pipeline_wave32 != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, gemv_pipeline_wave32, nullptr);
                gemv_pipeline_wave32 = VK_NULL_HANDLE;
            }
            if (gemm_pipeline_wave32 != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, gemm_pipeline_wave32, nullptr);
                gemm_pipeline_wave32 = VK_NULL_HANDLE;
            }
            if (gemv_pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, gemv_pipeline, nullptr);
                gemv_pipeline = VK_NULL_HANDLE;
            }
            if (gemm_pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, gemm_pipeline, nullptr);
                gemm_pipeline = VK_NULL_HANDLE;
            }
            if (gemv_shader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, gemv_shader, nullptr);
                gemv_shader = VK_NULL_HANDLE;
            }
            if (gemm_shader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, gemm_shader, nullptr);
                gemm_shader = VK_NULL_HANDLE;
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
                pipeline_layout = VK_NULL_HANDLE;
            }
            if (desc_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, desc_pool, nullptr);
                desc_pool = VK_NULL_HANDLE;
            }
            if (desc_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
                desc_layout = VK_NULL_HANDLE;
            }
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
                fence = VK_NULL_HANDLE;
            }
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
            }
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }

    Impl() {
        try {
            init_vulkan();
        } catch (const std::exception& e) {
            cleanup_vulkan();
            vulkan_initialized = false;
            init_error = e.what();
            device_name = "CPU Fallback (Vulkan Init Failed: " + init_error + ")";
            std::cerr << "[GpuBackend] Warning: Vulkan initialization failed (" << e.what()
                      << "). Using CPU fallback mode.\n";
        }
    }

    ~Impl() {
        cleanup_vulkan();
    }

    void init_vulkan() {
        // 1. Create Instance (request Vulkan 1.3 or 1.2)
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "alveare_runtime";
        app_info.applicationVersion = VK_MAKE_VERSION(3, 0, 0);
        app_info.pEngineName = "alveare_gpu_engine";
        app_info.engineVersion = VK_MAKE_VERSION(3, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo inst_info{};
        inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        inst_info.pApplicationInfo = &app_info;

        VkResult res = vkCreateInstance(&inst_info, nullptr, &instance);
        if (res != VK_SUCCESS) {
            app_info.apiVersion = VK_API_VERSION_1_2;
            res = vkCreateInstance(&inst_info, nullptr, &instance);
            if (res != VK_SUCCESS) {
                throw std::runtime_error("vkCreateInstance failed with code " + std::to_string(res));
            }
        }

        // 2. Pick Physical Device (prefer discrete or integrated GPU)
        uint32_t dev_count = 0;
        res = vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
        if (res != VK_SUCCESS || dev_count == 0) {
            throw std::runtime_error("No Vulkan physical devices found");
        }

        std::vector<VkPhysicalDevice> devices(dev_count);
        res = vkEnumeratePhysicalDevices(instance, &dev_count, devices.data());
        if (res != VK_SUCCESS) {
            throw std::runtime_error("vkEnumeratePhysicalDevices failed");
        }

        for (auto pd : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device = pd;
                device_name = std::string(props.deviceName) + " (Vulkan Native)";
                break;
            }
        }
        if (!physical_device && !devices.empty()) {
            physical_device = devices[0];
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_device, &props);
            device_name = std::string(props.deviceName) + " (Vulkan)";
        }

        // 3. Query physical device features: 16-bit storage, shaderInt16, subgroup control
        VkPhysicalDevice16BitStorageFeatures storage16{};
        storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

        VkPhysicalDeviceSubgroupSizeControlFeatures sscf{};
        sscf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
        storage16.pNext = &sscf;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &storage16;

        vkGetPhysicalDeviceFeatures2(physical_device, &features2);

        if (!features2.features.shaderInt16) {
            throw std::runtime_error("Vulkan device does not support required feature: shaderInt16");
        }
        if (!storage16.storageBuffer16BitAccess) {
            throw std::runtime_error("Vulkan device does not support required feature: storageBuffer16BitAccess");
        }

        // Query subgroup properties and subgroup size control support
        VkPhysicalDeviceSubgroupProperties subgroup_props{};
        subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceSubgroupSizeControlProperties sscp{};
        sscp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
        subgroup_props.pNext = &sscp;

        VkPhysicalDeviceProperties2 prop2{};
        prop2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        prop2.pNext = &subgroup_props;
        vkGetPhysicalDeviceProperties2(physical_device, &prop2);

        default_subgroup_sz = subgroup_props.subgroupSize;
        bool has_subgroup_control = (sscf.subgroupSizeControl == VK_TRUE) &&
                                    (sscp.minSubgroupSize <= 32 && sscp.maxSubgroupSize >= 32) &&
                                    ((sscp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0);
        supports_subgroup_control = has_subgroup_control;

        // 4. Find Compute Queue Family
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
        if (!found_queue) throw std::runtime_error("No compute queue family found on physical device");

        // 5. Create Logical Device with proper 16-bit storage features chain
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = compute_queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        VkPhysicalDevice16BitStorageFeatures enable_storage16{};
        enable_storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        enable_storage16.storageBuffer16BitAccess = VK_TRUE;

        VkPhysicalDeviceSubgroupSizeControlFeatures enable_sscf{};
        enable_sscf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;

        if (has_subgroup_control) {
            enable_sscf.subgroupSizeControl = VK_TRUE;
            enable_sscf.computeFullSubgroups = VK_TRUE;
            enable_storage16.pNext = &enable_sscf;
        }

        VkPhysicalDeviceFeatures2 enable_features2{};
        enable_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        enable_features2.features.shaderInt16 = VK_TRUE;
        enable_features2.pNext = &enable_storage16;

        VkDeviceCreateInfo dev_info{};
        dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dev_info.pNext = &enable_features2;
        dev_info.pEnabledFeatures = nullptr; // Required null when pNext has VkPhysicalDeviceFeatures2
        dev_info.queueCreateInfoCount = 1;
        dev_info.pQueueCreateInfos = &queue_info;

        res = vkCreateDevice(physical_device, &dev_info, nullptr, &device);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("vkCreateDevice failed with code " + std::to_string(res));
        }
        vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);

        // 6. Command Pool, Buffer & Fence
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = compute_queue_family;
        res = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateCommandPool failed");

        VkCommandBufferAllocateInfo cmd_info{};
        cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_info.commandPool = command_pool;
        cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_info.commandBufferCount = 1;
        res = vkAllocateCommandBuffers(device, &cmd_info, &command_buffer);
        if (res != VK_SUCCESS) throw std::runtime_error("vkAllocateCommandBuffers failed");

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        res = vkCreateFence(device, &fence_info, nullptr, &fence);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateFence failed");

        // 7. Descriptor Set Layout (3 SSBOs: weights, input, output)
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
        res = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &desc_layout);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateDescriptorSetLayout failed");

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
        res = vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreatePipelineLayout failed");

        // 8. Descriptor Pool & Set
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 32;

        VkDescriptorPoolCreateInfo dp_info{};
        dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp_info.flags = 0;
        dp_info.maxSets = 8;
        dp_info.poolSizeCount = 1;
        dp_info.pPoolSizes = &pool_size;
        res = vkCreateDescriptorPool(device, &dp_info, nullptr, &desc_pool);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateDescriptorPool failed");

        VkDescriptorSetAllocateInfo ds_info{};
        ds_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_info.descriptorPool = desc_pool;
        ds_info.descriptorSetCount = 1;
        ds_info.pSetLayouts = &desc_layout;
        res = vkAllocateDescriptorSets(device, &ds_info, &desc_set);
        if (res != VK_SUCCESS) throw std::runtime_error("vkAllocateDescriptorSets failed");

        // 9. Create Compute Shader Modules & Pipelines
        VkShaderModuleCreateInfo sm_info{};
        sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_info.codeSize = gpu_shaders::gemv_q4_0_spv_size;
        sm_info.pCode = gpu_shaders::gemv_q4_0_spv;
        res = vkCreateShaderModule(device, &sm_info, nullptr, &gemv_shader);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateShaderModule (GEMV) failed");

        VkComputePipelineCreateInfo cp_info{};
        cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_info.layout = pipeline_layout;
        cp_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp_info.stage.module = gemv_shader;
        cp_info.stage.pName = "main";
        res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &gemv_pipeline);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateComputePipelines (GEMV) failed");

        // Optional Wave32 GEMV pipeline
        if (has_subgroup_control) {
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req_sub32{};
            req_sub32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
            req_sub32.requiredSubgroupSize = 32;

            VkComputePipelineCreateInfo cp_info32 = cp_info;
            cp_info32.stage.pNext = &req_sub32;
            cp_info32.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info32, nullptr, &gemv_pipeline_wave32);
        }

        // GEMM pipeline
        sm_info.codeSize = gpu_shaders::gemm_q4_0_spv_size;
        sm_info.pCode = gpu_shaders::gemm_q4_0_spv;
        res = vkCreateShaderModule(device, &sm_info, nullptr, &gemm_shader);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateShaderModule (GEMM) failed");

        cp_info.stage.module = gemm_shader;
        res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &gemm_pipeline);
        if (res != VK_SUCCESS) throw std::runtime_error("vkCreateComputePipelines (GEMM) failed");

        // Optional Wave32 GEMM pipeline
        if (has_subgroup_control) {
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req_sub32{};
            req_sub32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
            req_sub32.requiredSubgroupSize = 32;

            VkComputePipelineCreateInfo cp_info32 = cp_info;
            cp_info32.stage.pNext = &req_sub32;
            cp_info32.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info32, nullptr, &gemm_pipeline_wave32);
        }

        // 10. Allocate persistent input/output activation buffers (initial 16MB)
        input_buffer = create_buffer(device, physical_device, 16 * 1024 * 1024,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        output_buffer = create_buffer(device, physical_device, 16 * 1024 * 1024,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        vulkan_initialized = true;
        std::cout << "[GpuBackend] Successfully initialized " << device_name
                  << " (default subgroup=" << default_subgroup_sz
                  << ", subgroup control=" << (has_subgroup_control ? "supported" : "not supported")
                  << ")!\n";
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

void GpuBackend::reset_profile() {
    elapsed_seconds_ = 0.0;
    ffn_seconds_ = 0.0;
    total_calls_ = 0;
    if (impl_) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stats = GpuStats{};
        impl_->last_logged_fallback_reason.clear();
    }
}

GpuBackend::GpuStats GpuBackend::stats() const {
    if (!impl_) return GpuStats{};
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats;
}

uint64_t GpuBackend::fallback_count() const {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats.fault_fallback_calls;
}

std::string GpuBackend::last_fallback_reason() const {
    if (!impl_) return "Backend unallocated";
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats.last_fallback_reason;
}

bool GpuBackend::supports_subgroup_size_control() const {
    return impl_ && impl_->supports_subgroup_control;
}

uint32_t GpuBackend::default_subgroup_size() const {
    return impl_ ? impl_->default_subgroup_sz : 0;
}

bool GpuBackend::enable_forced_subgroup32(bool enable) {
    if (!impl_ || !impl_->supports_subgroup_control || !impl_->gemv_pipeline_wave32) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->force_wave32 = enable;
    return true;
}

bool GpuBackend::is_forced_subgroup32() const {
    return impl_ && impl_->force_wave32;
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
            impl_->record_fault_fallback("Weight buffer creation failed: " + std::string(e.what()));
        }
    }
    impl_->weights.push_back(std::move(gw));
    return handle;
}

WeightHandle GpuBackend::create_ffn_fused_weight(int H, int I, const std::string& activation,
                                                 const void* packed, size_t nbytes) {
    // Registered in host fallback/internal structure (intentional CPU FFN path)
    return impl_->cpu_fallback.create_ffn_fused_weight(H, I, activation, packed, nbytes);
}

void GpuBackend::run_gemv(int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) {
    auto t0 = std::chrono::steady_clock::now();

    if (!impl_->vulkan_initialized) {
        impl_->record_fault_fallback("Vulkan not initialized");
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    if (N <= 0 || K <= 0 || (K % 32) != 0) {
        impl_->record_fault_fallback("Invalid GEMV dimensions: N=" + std::to_string(N) + ", K=" + std::to_string(K) + " (K must be positive multiple of 32)");
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    if (w >= impl_->weights.size() || !impl_->weights[w].buffer.buffer) {
        impl_->record_fault_fallback("Weight handle invalid or GPU weight buffer unallocated");
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    if (x_bf16 == nullptr || y_bf16 == nullptr) {
        impl_->record_fault_fallback("Null activation pointer passed to run_gemv");
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto& gw = impl_->weights[w];

    size_t in_bytes = static_cast<size_t>(K) * sizeof(bf16);
    size_t out_bytes = static_cast<size_t>(N) * sizeof(bf16);

    try {
        impl_->ensure_buffer_capacity(impl_->input_buffer, in_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        impl_->ensure_buffer_capacity(impl_->output_buffer, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } catch (const std::exception& e) {
        impl_->record_fault_fallback("Buffer reallocation failed: " + std::string(e.what()));
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    // 1. Copy activation to mapped memory
    auto t_copy_in = std::chrono::steady_clock::now();
    std::memcpy(impl_->input_buffer.mapped, x_bf16, in_bytes);
    impl_->stats.copy_in_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_copy_in).count();

    // 2. Update Descriptors: 0: weight, 1: input, 2: output
    VkDescriptorBufferInfo buf_infos[3]{};
    buf_infos[0].buffer = gw.buffer.buffer;
    buf_infos[0].offset = 0;
    buf_infos[0].range = gw.buffer.size;

    buf_infos[1].buffer = impl_->input_buffer.buffer;
    buf_infos[1].offset = 0;
    buf_infos[1].range = in_bytes;

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
    VkResult res = vkBeginCommandBuffer(impl_->command_buffer, &begin_info);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkBeginCommandBuffer failed with code " + std::to_string(res));
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    VkPipeline active_pipeline = (impl_->force_wave32 && impl_->gemv_pipeline_wave32)
                                 ? impl_->gemv_pipeline_wave32
                                 : impl_->gemv_pipeline;

    vkCmdBindPipeline(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, active_pipeline);
    vkCmdBindDescriptorSets(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            impl_->pipeline_layout, 0, 1, &impl_->desc_set, 0, nullptr);

    int pc_data[2] = { N, K };
    vkCmdPushConstants(impl_->command_buffer, impl_->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_data), pc_data);

    vkCmdDispatch(impl_->command_buffer, static_cast<uint32_t>(N), 1, 1);
    res = vkEndCommandBuffer(impl_->command_buffer);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkEndCommandBuffer failed with code " + std::to_string(res));
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &impl_->command_buffer;

    res = vkResetFences(impl_->device, 1, &impl_->fence);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkResetFences failed with code " + std::to_string(res));
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    res = vkQueueSubmit(impl_->compute_queue, 1, &submit_info, impl_->fence);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkQueueSubmit failed with code " + std::to_string(res));
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    // Bounded fence wait (30 seconds)
    const uint64_t timeout_ns = 30000000000ULL;
    auto t_fence_start = std::chrono::steady_clock::now();
    res = vkWaitForFences(impl_->device, 1, &impl_->fence, VK_TRUE, timeout_ns);
    double fence_dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fence_start).count();
    impl_->stats.gpu_fence_wait_seconds += fence_dt;

    if (res != VK_SUCCESS) {
        std::string err = (res == VK_TIMEOUT) ? "GPU fence wait timeout (30s) - potential GPU hang"
                                              : ("vkWaitForFences failed with code " + std::to_string(res));
        impl_->record_fault_fallback(err);
        impl_->cpu_fallback.run_gemv(N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    // 4. Read output back
    auto t_copy_out = std::chrono::steady_clock::now();
    std::memcpy(y_bf16, impl_->output_buffer.mapped, out_bytes);
    impl_->stats.copy_out_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_copy_out).count();

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    elapsed_seconds_ += dt;
    impl_->stats.wall_seconds += dt;
    ++total_calls_;
    ++impl_->stats.gemv_gpu_calls;
}

void GpuBackend::run_gemv_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                const void* x_bf16, void* y_bf16_concat) {
    bf16* out_ptr = static_cast<bf16*>(y_bf16_concat);
    for (size_t i = 0; i < weights.size(); ++i) {
        run_gemv(N, K, weights[i], x_bf16, out_ptr + i * N);
    }
    // Note: run_gemv already accurately increments total_calls_ (per projection)
    // and elapsed_seconds_. Double-counting fix: do NOT add outer loop duration again!
}

void GpuBackend::run_gemv_multi_in_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                         const std::vector<const void*>& x_ptrs, void* y_bf16_concat) {
    bf16* out_ptr = static_cast<bf16*>(y_bf16_concat);
    for (size_t i = 0; i < weights.size(); ++i) {
        const void* x_in = (i < x_ptrs.size()) ? x_ptrs[i] : x_ptrs[0];
        run_gemv(N, K, weights[i], x_in, out_ptr + i * N);
    }
    // Double-counting fix: do NOT add outer loop duration again!
}

void GpuBackend::run_gemm(int B, int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) {
    auto t0 = std::chrono::steady_clock::now();

    if (!impl_->vulkan_initialized || !impl_->gemm_pipeline) {
        impl_->record_fault_fallback("Vulkan GEMM pipeline unavailable");
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    if (B <= 0 || N <= 0 || K <= 0 || (K % 32) != 0) {
        impl_->record_fault_fallback("Invalid GEMM dimensions: B=" + std::to_string(B) + ", N=" + std::to_string(N) + ", K=" + std::to_string(K));
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    if (w >= impl_->weights.size() || !impl_->weights[w].buffer.buffer) {
        impl_->record_fault_fallback("Invalid weight handle or unallocated weight buffer in run_gemm");
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    if (x_bf16 == nullptr || y_bf16 == nullptr) {
        impl_->record_fault_fallback("Null buffer pointer passed to run_gemm");
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto& gw = impl_->weights[w];

    size_t in_bytes = static_cast<size_t>(B) * K * sizeof(bf16);
    size_t out_bytes = static_cast<size_t>(B) * N * sizeof(bf16);

    try {
        impl_->ensure_buffer_capacity(impl_->input_buffer, in_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        impl_->ensure_buffer_capacity(impl_->output_buffer, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } catch (const std::exception& e) {
        impl_->record_fault_fallback("GEMM buffer reallocation failed: " + std::string(e.what()));
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    auto t_copy_in = std::chrono::steady_clock::now();
    std::memcpy(impl_->input_buffer.mapped, x_bf16, in_bytes);
    impl_->stats.copy_in_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_copy_in).count();

    VkDescriptorBufferInfo buf_infos[3]{};
    buf_infos[0].buffer = gw.buffer.buffer;
    buf_infos[0].offset = 0;
    buf_infos[0].range = gw.buffer.size;

    buf_infos[1].buffer = impl_->input_buffer.buffer;
    buf_infos[1].offset = 0;
    buf_infos[1].range = in_bytes;

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
    VkResult res = vkBeginCommandBuffer(impl_->command_buffer, &begin_info);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkBeginCommandBuffer failed in run_gemm");
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    VkPipeline active_pipeline = (impl_->force_wave32 && impl_->gemm_pipeline_wave32)
                                 ? impl_->gemm_pipeline_wave32
                                 : impl_->gemm_pipeline;

    vkCmdBindPipeline(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, active_pipeline);
    vkCmdBindDescriptorSets(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            impl_->pipeline_layout, 0, 1, &impl_->desc_set, 0, nullptr);

    int pc_data[3] = { B, N, K };
    vkCmdPushConstants(impl_->command_buffer, impl_->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_data), pc_data);

    vkCmdDispatch(impl_->command_buffer, static_cast<uint32_t>(N), static_cast<uint32_t>(B), 1);
    res = vkEndCommandBuffer(impl_->command_buffer);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkEndCommandBuffer failed in run_gemm");
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &impl_->command_buffer;

    res = vkResetFences(impl_->device, 1, &impl_->fence);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkResetFences failed in run_gemm");
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    res = vkQueueSubmit(impl_->compute_queue, 1, &submit_info, impl_->fence);
    if (res != VK_SUCCESS) {
        impl_->record_fault_fallback("vkQueueSubmit failed in run_gemm with code " + std::to_string(res));
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    const uint64_t timeout_ns = 30000000000ULL;
    auto t_fence_start = std::chrono::steady_clock::now();
    res = vkWaitForFences(impl_->device, 1, &impl_->fence, VK_TRUE, timeout_ns);
    double fence_dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fence_start).count();
    impl_->stats.gpu_fence_wait_seconds += fence_dt;

    if (res != VK_SUCCESS) {
        std::string err = (res == VK_TIMEOUT) ? "GPU fence wait timeout (30s) in GEMM"
                                              : ("vkWaitForFences failed with code " + std::to_string(res));
        impl_->record_fault_fallback(err);
        impl_->cpu_fallback.run_gemm(B, N, K, w, x_bf16, y_bf16);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        elapsed_seconds_ += dt;
        ++total_calls_;
        return;
    }

    auto t_copy_out = std::chrono::steady_clock::now();
    std::memcpy(y_bf16, impl_->output_buffer.mapped, out_bytes);
    impl_->stats.copy_out_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_copy_out).count();

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    elapsed_seconds_ += dt;
    impl_->stats.wall_seconds += dt;
    ++total_calls_;
    ++impl_->stats.gemm_gpu_calls;
}

void GpuBackend::run_gemm_streamed(int B, int N, int K, const void* packed, size_t nbytes,
                                   const void* x_bf16, void* y_bf16) {
    // Streamed host-weights delegation to multi-threaded CPU fallback
    impl_->cpu_fallback.run_gemm_streamed(B, N, K, packed, nbytes, x_bf16, y_bf16);
}

void GpuBackend::run_ffn_fused(int H, int I, const std::string& activation, WeightHandle w,
                               const void* x_bf16, void* y_bf16) {
    // Intentional CPU FFN execution path (tracked distinctly from fault fallbacks)
    auto t0 = std::chrono::steady_clock::now();
    impl_->cpu_fallback.run_ffn_fused(H, I, activation, w, x_bf16, y_bf16);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ffn_seconds_ += dt;
    elapsed_seconds_ += dt;
    ++total_calls_;
    ++impl_->stats.intentional_cpu_ffn_calls;
}

} // namespace alveare
