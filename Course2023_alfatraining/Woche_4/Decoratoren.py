## Decorators in functions
def myDecorator(fkt):
    def FktWrapper():
        print("Before function call")
        fkt()
        print("After function call")
    return FktWrapper


@myDecorator
def myFkt():
    print("Inside the function")


myFkt()
print("\n")


## Decorators in classes
def myDecorator(aclass):
    class myWrapper:
        def __init__(self, *args, **kwargs):
            self.wrapped = aclass(*args, **kwargs)

        def decorated_method(self):
            print("Before method call")
            self.wrapped.original_method()
            print("After method call")

    return myWrapper


@myDecorator
class myClass:
    def __init__(self, name):
        self.name = name

    def original_method(self):
        print(f"Hello, {self.name}!")


test = myClass('Shiva')
test.decorated_method()
