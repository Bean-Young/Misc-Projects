
def test1():
    import math

    class Vector:
        def __init__(self, x=0, y=0, z=0):
            self.x = x
            self.y = y
            self.z = z

        def __add__(self, other):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

        def __sub__(self, other):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

        def __mul__(self, scalar):
            return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

        def __truediv__(self, scalar):
            return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

        def __abs__(self):
            return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)
    v3 = v1 + v2
    print(v3.x, v3.y, v3.z)  # 5 7 9
    v4 = Vector(1, 1, 1)
    v5 = v1 - v4
    print(v5.x, v5.y, v5.z)  # 0 1 2
    v6 = v1 * 2
    print(v6.x, v6.y, v6.z)  # 2 4 6
    v7 = v1 / 2
    print(v7.x, v7.y, v7.z)  # 0.5 1.0 1.5
    length = abs(v1)
    print(length)  # 3.7416573867739413

def test2():

    import time

    class Queue:
        def __init__(self, size=20):
            self._content = []
            self._size = size
            self._current = 0

        def setSize(self, size):
            if size < self._current:
                for i in range(size, self._current)[::-1]:
                    del self._content[i]
                self._current = size
            self._size = size

        def put(self, v, timeout=9):

            if self._current < self._size:
                self._content.append(v)
                self._current = self._current + 1
            else:
                for i in range(timeout):
                    time.sleep(1)
                    if self._current < self._size:
                        self._content.append(v)
                        self._current = self._current + 1
                        break
                else:
                    return '队列已满，超时放弃'

        def get(self, timeout=9):

            if self._content:
                self._current = self._current - 1
                return self._content.pop(0)
            else:
                for i in range(timeout):
                    time.sleep(1)
                if self._content:
                    self._current = self._current - 1
                    return self._content.pop(0)
                else:
                    return '队列为空，超时放弃'
        def show(self):

            if self._content:
                print(self._content)
            else:
                print('The queue is empty')
        def empty(self):
            self._content = []
            self._current = 0

        def isEmpty(self):
            return not self._content


        def isFull(self):
            return self._current == self._size

    q = Queue(5)
    q.put(1)
    q.put(2)
    q.put(3)
    q.show()  # [1, 2, 3]
    print(q.get())
    q.show()  # [2, 3]
    print(q.isEmpty())  # False
    print(q.isFull())  # False
    q.put(4)
    q.put(5)
    q.put(6)
    print(q.put(7)) # 队列已满，超时放弃
    q.show() # [2,3,4,5,6]
    q.empty()
    q.show()  # The queue is empty


if __name__=='__main__':
    test1()
    test2()
    
