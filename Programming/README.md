<img src="../img/fotis-fotopoulos-6sAl6aQ4OWI-unsplash.jpg" width=100% />

<div align=center><h4>Programming Skills</h4></div>

- [ ] [Regex]()
- [ ] [Linux System Commands]()
- [ ] [Shell Script]()
- [ ] [Python Decoration Function]()
- [ ] [Basic Web Scrapping]()
- [x] Classmethod vs. Staticmethod
    - A class method takes cls as the first parameter while a static method needs no specific parameters.
    - A class method can access or modify the class state while a static method can’t access or modify it.
    - In general, static methods know nothing about the class state. They are utility-type methods that take some parameters and work upon those parameters. On the other hand class methods must have class as a parameter.
    - We use @classmethod decorator in python to create a class method and we use @staticmethod decorator to create a static method in python.
    
    ```python
    # example of use of classmehtod and staticmethod
    class Distance:
        # a static method calculate the minkowski distance based on given array X, Y
        @staticmethod
        def Minkowski(X, Y, p):
            return np.power(np.sum(np.abs(X - Y)), (1 / p))

        # a class method calculate the Manhatten distance based on Distance object method
        @classmethod
        def Manhatten(clf, X, Y):
            return clf.Minkowski(X, Y, p=1)
    ```
    - [ ] [MySQL]()
    - [ ] [Sqlite]()
    - [ ] [MongoDB]()