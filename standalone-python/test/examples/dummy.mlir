// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        // CHECK: %{{.*}} = python.undefined : !python.value
        %res = python.undefined : !python.value
        return
    }
}
