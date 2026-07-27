#include "alveare/prompt_lookup.h"
#include <cassert>
#include <iostream>
#include <vector>

void test_prompt_lookup() {
    // Case 1: pattern [1, 2, 3] matched at i=0, 1 token available before suffix [1, 2, 3] at n-g
    auto res1 = alveare::propose_draft({1, 2, 3, 4, 1, 2, 3}, 4, 3, 2);
    assert((res1 == std::vector<int>{4}));

    // Case 2: pattern [5, 6] (g=2) matched at i=0, tokens following match up to suffix
    auto res2 = alveare::propose_draft({5, 6, 7, 8, 9, 5, 6}, 4, 3, 2);
    assert((res2 == std::vector<int>{7, 8, 9}) || (res2 == std::vector<int>{7, 8, 9, 5}));

    // Case 3: No repetition in tokens
    auto res3 = alveare::propose_draft({1, 2, 3}, 4, 3, 2);
    assert(res3.empty());

    // Case 4: Empty sequence or single element
    assert(alveare::propose_draft({}, 4, 3, 2).empty());
    assert(alveare::propose_draft({42}, 4, 3, 2).empty());

    std::cout << "All prompt lookup tests passed successfully!\n";
}

int main() {
    test_prompt_lookup();
    return 0;
}
