// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = python.foo %{{.*}} : i32
        %res = python.foo %0 : i32
        return
    }
}
