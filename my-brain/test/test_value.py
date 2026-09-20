from brain.autograd import Value

a = Value(2.0)
b = Value(3.0)

c = a * b

d = c + a

print("c = ",c)
print("d = ",d)


a = Value(3.3)
b = Value(2.7)

c = a * b
d = c + a

print("a = ",a)
print("b = ",b)
print("c = ",c)
print("d = ",d)

d.backward()

print(a.grad)
print(b.grad)
