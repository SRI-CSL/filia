// RUN: script-opt %s | script-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        // CHECK: %{{.*}} = python.undefined : !python.value
        %res = python.undefined : !python.value
        return
    }
}
