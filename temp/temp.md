```python

class GradeMap:
    _instance = None

    @classmethod
    def _getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls._getInstance
        return cls._instance

    def __init__(self):
        self._dict = {}

    def setitem(self, key, item):
        self._dict[key] = item
```

```python
from student.student_utilities.grade_map import GradeMap

class Student:
    def __init__(self, age, name, korean, mathematics, english):
        self._age = age
        self._grade = GradeMap.instance().get_grade(age)

    @property
    def age(self):
        return self._age

    @property
    def grade(self):
        return self._grade
```