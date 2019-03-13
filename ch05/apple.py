"""
计算苹果总价
"""
from layer import MulLayer



if __name__ =="__main__":
    """
    """
    apple = 100
    apple_num = 2
    tax = 1.1

    apple_layer = MulLayer()
    tax_layer = MulLayer()

    # forward
    apple_price = apple_layer.forward(apple,apple_num)
    sum_price = tax_layer.forward(apple_price,tax)

    print("apple_price = %d ;sum_price = %d"%(apple_price,sum_price))

    # backward
    dprice = 1
    # 顺序相反
    dapple_price,dtax = tax_layer.backward(dprice)
    dapple,dapple_num = apple_layer.backward(dapple_price)
    print("dtax =%f ; dapple_price =%f ;dapple =%f ; dapple_num =%f "%(dtax,dapple_price,dapple,dapple_num))
