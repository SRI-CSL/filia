def f():
        def h():
                {
                        x := 3
                }
                print("Hello from h")
                print(x)
        x:=2
        g()
        h()

def g():
        print("Hello from g")
        print(x)

x=1
f()