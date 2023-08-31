# UnitsPythonPackage
Install with: 
```
pip install units-python
```

### How to use:

In this example, we have a physics problem to solve. 

We will solve it using units-python to keep track of our units:

**Physics question**:
How many soccer balls (diameter of 22 cm) can fit inside a 5 x 5 x 3 meter room?

```
from units-python import v
import units-python as up

soccer_ball_diameter = v("22 cm")
soccer_ball_radius = soccer_ball_diameter / 2
soccer_ball_volume = 4/3 * up.pi * soccer_ball_radius ** 3

room_length = v("5 m")
room_width = v("5 m")
room_height = v("5 m")
room_volume = room_length * room_width * room_height

number_of_soccer_balls_in_room = room_volume / soccer_ball_volume
print(number_of_soccer_balls_in_room)
```

### Custom functions:
If you want to use either of those, import them from units-python:
```
from units-python import v
import units-python as up

my_value = v("9 m^2")
other_value = v("21 m^3")

my_value_sqrt = up.sqrt(my_value)
print(my_value_sqrt) # outputs "3 m"

my_value_3sqrt = up.nsqrt(my_value, 3)
print(my_value_sqrt) # outputs "3 m"
```


### What you, at this time, cannot do:
- sqrt
- exp
- value ** value (value ** constant works!)
