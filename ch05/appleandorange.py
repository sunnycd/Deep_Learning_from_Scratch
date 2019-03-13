"""
苹果和橙子总价
"""

from layer import MulLayer
from layer import AddLayer

if __name__ == "__main__":
    """
    """
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    apple_layer = MulLayer()
    orange_layer = MulLayer()
    add_apple_orange = AddLayer()
    tax_layer = MulLayer()

    # forward
    apple_price = apple_layer.forward(apple,apple_num)
    orange_price = orange_layer.forward(orange,orange_num)
    add_price = add_apple_orange.forward(apple_price,orange_price)
    sum_price = tax_layer.forward(add_price,tax)

    # backward
    dprice = 1
    dadd_price,dtax = tax_layer.backward(dprice)
    dapple_price,dorange_price = add_apple_orange.backward(dadd_price)
    dapple,dapp_num = apple_layer.backward(dapple_price)
    dorange,dorange_num = orange_layer.backward(dorange_price)

    print("sum_price = ",sum_price//1)
    print("dtax =%f ; dadd_price =%f ; dapple_price =%f ; dorange_price =%f ;dapple =%f ;dapp_num =%f ;\
    dorange =%f;dorange_num =%f"%(dtax,dadd_price,dapple_price,dorange_price,dapple,dapp_num,dorange,dorange_num))