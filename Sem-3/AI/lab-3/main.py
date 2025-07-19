"""Create a Class with instance attributes
Write a Python program to create a Vehicle class with max_speed and mileage instance
attributes."""

"""class Vehicle:
 def __init__(self, max_speed, mileage):
  self.max_speed = max_speed
  self.mileage = mileage

v = Vehicle(140, 20)

print(v.max_speed, "\n", v.mileage)"""


#Create a Vehicle class without any variables and methods

"""class Vehicle:
 pass"""


"""Create a child class Bus that will inherit all of the
variables and methods of the Vehicle class"""

"""class Vehicle:
 def __init__(self, max_speed, mileage):
  self.max_speed = max_speed
  self.mileage = mileage

class Bus(Vehicle):
 pass

b = Bus(180, 17)

print(b.max_speed, "\n", b.mileage)"""


"""Class Inheritance
Given:
Create a Bus class that inherits from the Vehicle class. Give the capacity argument
of Bus.seating_capacity() a default value of 50."""

"""class Vehicle:
 def __init__(self, max_speed, mileage):
  self.max_speed = max_speed
  self.mileage = mileage

class Bus(Vehicle):
 def __init__(self, max_speed, mileage, seating_capacity = 50):
  super().__init__(max_speed, mileage)
  self.seating_capacity = seating_capacity

b = Bus(180, 17)
print(b.max_speed, b.mileage, b.seating_capacity)"""


"""Define a property that must have the same value for every
class instance (object)
Define a class attribute ”color” with a default value white. I.e., Every Vehicle should be
white."""

"""class vehicle:
 color= "white"
 def __init__(self, make, model):
     self.make=make
     self.model=model

v1=vehicle("Toyota", "2019")
v2=vehicle("Audi" , "2020")
print(v1.make, v1.model, v1.color)
print(v2.make, v2.model, v2.color)"""


"""Class Inheritance
Given:
Create a Bus child class that inherits from the Vehicle class. The default fare charge of
any vehicle is seating capacity * 100. If Vehicle is Bus instance, we need to add an extra
10% on full fare as a maintenance charge. So total fare for bus instance will become
the final amount = total fare + 10% of the total fare.
Note: The bus seating capacity is 50. so the final fare amount should be 5500. You need
to override the fare() method of a Vehicle class in Bus class."""

"""class Vehicle:
    def __init__(self, seating_capacity):
        self.seating_capacity = seating_capacity

    def fare(self):
        return self.seating_capacity * 100

class Bus(Vehicle):
    def fare(self):
        total_fare = super().fare()
        maintenance_charge = total_fare * 
        final_fare = total_fare + maintenance_charge
        return final_fare

bus = Bus(50)
fare = bus.fare()
print(fare)
"""