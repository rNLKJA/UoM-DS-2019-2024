<img src="https://www.interviewbit.com/blog/wp-content/uploads/2021/08/dbms.jpg" width=100% />
    
<div align=center><h2>Database Management System <a href='https://www.appdynamics.com/topics/database-management-systems'>(ADDYNAMICS)</a></h2></div>


Database Management Systems (DBMS) are software systems used to store, retrieve, and run queries on data. A DBMS serves as an interface between an end-user and a database, allowing users to create, read, update, and delete data in the database.

DBMS manages the data, the database engine, and the database schema, allowing for data to be manipulated or extracted by users and other programs. This helps provide data security, data integrity, concurrency, and uniform data administration procedures.

DBMS optimizes the organization of data by following a database schema design technique called normalization, which splits a large table into smaller tables when any of its attributes have redundancy in values. DBMS offers many benefits over traditional file systems, including flexibility and a more complex backup system.

Database management systems can be classified based on a variety of criteria such as the data model, the database distribution, or user numbers. The most widely used types of DBMS software are relational, distributed, hierarchical, object-oriented, and network.

**Distributed database management system**: A distributed DBMS is a set of logically interrelated databases distributed over a network that is managed by a centralized database application. This type of DBMS synchronizes data periodically and ensures that any data change is universally updated in the database.

**Hierarchical database management system**: Hierarchical databases organize model data in a tree-like structure. Data storage is either a top-down or bottom-up format and is represented using a parent-child relationship.

**Network database management system**: The network database model addresses the need for more complex relationships by allowing each child to have multiple parents. Entities are organized in a graph that can be accessed through several paths.

**Relational database management system**: Relational database management systems (RDBMS) are the most popular data model because of their user-friendly interface. It is based on normalizing data in the rows and columns of the tables. This is a viable option when you need a data storage system that is scalable, flexible, and able to manage lots of information.

**Object-oriented database management system**: Object-oriented models store data in objects instead of rows and columns. It is based on object-oriented programming (OOP) that allows objects to have members such as fields, properties, and methods.

## Data Modeling

### ER Model
- ER model stands for Entity-Relationship Model. It is a high-level data model. This model is used to define the data elements and relationships for a specified system.
- It develops a conceptual design for the database. It also develops a very simple and easy design view of data.
- In ER modelling, the database structure is portrayed as a diagram called an entity-relationship diagram (ERD).
- ERD is composed of three elements: entity, attribute, and relation.

#### Entity
An entity may be an object, person or place. In the ER diagram, an entity can be represented as rectangles. Consider an organisation as an example - manager, product, employee, department, etc. can be taken as an entity.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept2.png" align=center width=100% />

In addition, an entity that depends on another entity is called a *weak entity*. The weak entity doesn't contain any key attribute of its own. The weak entity is represented by a double rectangle.

> DIFFERENT NOTATION STYLES MAY HAVE DIFFERENT REPRESENTATIONS

#### Attributes
The attribute is used to describe the property of an entity. Eclipse is used to represent an attribute.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept4.png" align=center width=100% />

##### Key Attribute
The key attribute is used to represent the main characteristics of an entity. It represents a primary key. The key attribute is represented by an ellipse with the text underlined.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept5.png" align=center width=100% />

##### Composite Attribute
An attribute composed of many other attributes is known as a composite attribute. The composite attribute is represented by an ellipse, and those ellipses are connected with an ellipse.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept6.png" align=center width=100% />

##### Multivaled Attribute
An attribute can have more than one value. These attributes are known as multivalued attributes. The double oval is used to represent multivalued attributes.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept7.png" align=center width=100% />

##### Derived Attribute
An attribute that can be derived from another attribute is known as a derived attribute. It can be represented by a dashed ellipse.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept8.png" align=center width=100% />

---

#### Relationship
A relationship is used to describe the relation between entities. A diamond or rhombus is used to represent the relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept9.png" align=center width=100% />

##### One to One Relationship
When only one instance of an entity is associated with the relationship, then it is known as one to one relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept10.png" align=center width=100% />

##### One to Many Relationship
When only one instance of the entity on the left, and more than one instance of an entity on the right is associated with the relationship then this is known as a one-to-many relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept11.png" align=center width=100% />

##### Many to One Relationship
When more than one instance of the entity on the left, and only one instance of an entity on the right are associated with the relationship then it is known as a many-to-one relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept12.png" align=center width=100% />

##### Many to Many Relationship
When more than one instance of the entity on the left, and more than one instance of an entity on the right are associated with the relationship then it is known as a many-to-many relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept13.png" align=center width=100% />

### Notation of ERD
The database can be represented using the notations. In ER diagram, many notations are used to express the cardinality. These notations are as follows:

<img src="https://static.javatpoint.com/dbms/images/dbms-notation-of-er-diagram.png" align=center width=100% />

---

#### ER Design Issue
##### Use of Entity Set vs Attributes
The use of entity set or attribute depends on the structure of the real-world enterprise that is being modelled and the semantics associated with its attributes. It leads to a mistake when the user uses the primary key of an entity set as an attribute of another entity set. Instead, he should use the relationship to do so. Also, the primary key attributes are implicit in the relationship set, but we designate them in the relationship sets.

##### Use of Entity Set vs. Relationship Sets
It is difficult to examine if an object can be best expressed by an entity set or relationship set. To understand and determine the right use, the user needs to designate a relationship set for describing an action that occurs between the entities. If there is a requirement of representing the object as a relationship set, then it's better not to mix it with the entity set.

##### Use of Binary vs n-ary Relationship Sets
Generally, the relationships described in the databases are binary. However, non-binary relationships can be represented by several binary relationships. For example, we can create and represent a ternary relationship 'parent' that may relate to a child, his father, as well as his mother. Such a relationship can also be represented by two binary relationships i.e, mother and father, that may relate to their child. Thus, it is possible to represent a non-binary relationship by a set of distinct binary relationships.

##### Placing Relationship Attributes
Generally, the relationships described in the databases are binary. However, non-binary relationships can be represented by several binary relationships. For example, we can create and represent a ternary relationship 'parent' that may relate to a child, his father, as well as his mother. Such a relationship can also be represented by two binary relationships i.e, mother and father, that may relate to their child. Thus, it is possible to represent a non-binary relationship by a set of distinct binary relationships.

---

#### Mapping constraints
- A mapping constraint is a data constraint that expresses the number of entities to which another entity can be related via a relationship set.
- It is most useful in describing the relationship sets that involve more than two entity sets.
- For binary relationship set R on an entity set A and B, there are four possible mapping cardinalities. These are as follows:

##### One-to-One
In one-to-one mapping, an entity in E1 is associated with at most one entity in E2, and an entity in E2 is associated with at most one entity in E1.

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints.png" align=center width=100% />

##### One-to-Many
In a one-to-many mapping, an entity in E1 is associated with any number of entities in E2, and an entity in E2 is associated with at most one entity in E1.

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints2.png" align=center width=100% />

##### Many-to-One
In a one-to-many mapping, an entity in E1 is associated with at most one entity in E2, and an entity in E2 is associated with any number of entities in E1.

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints3.png" align=center width=100% />

##### Many-to-Many
In many-to-many mapping, an entity in E1 is associated with any number of entities in E2, and an entity in E2 is associated with any number of entities in E1.

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints4.png" align=center width=100% />

---

#### DBMS Keys
- Keys play an important role in the relational database.
- It is used to uniquely identify any record or row of data from the table. It is also used to establish and identify relationships between tables.
- There are 7 types of keys:

##### Primary Key
- It is the first key used to identify one and only one instance of an entity uniquely

##### Candidate Key
- A candidate key is an attribute or set of attributes that can uniquely identify a tuple.
- Except for the primary key, the remaining attributes are considered candidate keys. The candidate keys are as strong as the primary key.

##### Super Key
- Super key is an attribute set that can uniquely identify a tuple. A super key is a superset of a candidate key.

##### Foreign Key
- Foreign keys are the column of the table used to point to the primary key of another table.

##### Alternate Key
- There may be one or more attributes or a combination of attributes that uniquely identify each tuple in a relation. These attributes or combinations of the attributes are called the candidate keys. One key is chosen as the primary key from these candidate keys, and the remaining candidate key, if it exists, is termed the alternate key. In other words, the total number of the alternate keys is the total number of candidate keys minus the primary key. The alternate key may or may not exist. If there is only one candidate key in a relation, it does not have an alternate key.

##### Composite Key
- Whenever a primary key consists of more than one attribute, it is known as a composite key. This key is also known as Concatenated Key.

##### Artificial Key
- The key created using arbitrarily assigned data are known as artificial keys. These keys are created when a primary key is large and complex and has no relationship with many other relations. The data values of the artificial keys are usually numbered in a serial order.

---

#### DBMS Generalisation
- Generalization is like a bottom-up approach in which two or more entities of the lower-level combine to form a higher-level entity if they have some attributes in common.
- In generalization, an entity of a higher level can also combine with the entities of the lower level to form a further higher-level entity.
- Generalization is more like a subclass and superclass system, but the only difference is the approach. Generalization uses the bottom-up approach.
- In generalization, entities are combined to form a more generalized entity, i.e., subclasses are combined to make a superclass.

---

#### DBMS Specialisation
- A specialization is a top-down approach, and it is the opposite of Generalization. In specialization, one higher-level entity can be broken down into two lower-level entities.
- Specialization is used to identify the subset of an entity set that shares some distinguishing characteristics.
- Normally, the superclass is defined first, the subclass and its related attributes are defined next, and the relationship sets are then added.- 

---

#### DBMS Aggregation
In aggregation, the relation between two entities is treated as a single entity. In aggregation, the relationship with its corresponding entities is aggregated into a higher-level entity.

---

#### Convert ER into table
The database can be represented using the notations, and these notations can be reduced to a collection of tables. In the database, every entity set or relationship set can be represented in tabular form.

<img src="https://static.javatpoint.com/dbms/images/dbms-reduction-of-er-diagram-into-table.png" align=center width=100% />

There are some points for converting the ER diagram to the table:
- Entity
    
    In the given ER diagram, LECTURE, STUDENT, SUBJECT and COURSE forms individual tables.
    
- All single-valued attribute becomes a column for the table.

    In the STUDENT entity, STUDENT_NAME and STUDENT_ID form the column of the STUDENT table. Similarly, COURSE_NAME and COURSE_ID form the column of the COURSE table and so on.
    
- A key attribute of the entity type is represented by the primary key.

    In the given ER diagram, COURSE_ID, STUDENT_ID, SUBJECT_ID, and LECTURE_ID are the key attribute of the entity.
    
- The multivalued attribute is represented by a separate table.

    At the student table, a hobby is a multivalued attribute. So it is not possible to represent multiple values in a single column of the STUDENT table. Hence we create a table STUD_HOBBY with column names STUDENT_ID and HOBBY. Using both columns, we create a composite key.
    
- Composite attribute represented by components.

    In the given ER diagram, student address is a composite attribute. It contains CITY, PIN, DOOR#, STREET, and STATE. In the STUDENT table, these attributes can merge as individual columns.
    
- Derived attributes are not considered in the table.
    
    In the STUDENT table, Age is the derived attribute. It can be calculated at any point in time by calculating the difference between the current date and Date of Birth.

Using these rules, you can convert the ER diagram to tables and columns and assign the mapping between the tables. The table structure for the given ER diagram is as below:

<img src="https://static.javatpoint.com/dbms/images/dbms-reduction-of-er-diagram-into-table2.png" align=center width=100% />

---

#### Relationship of Higher Degree
The degree of relationship can be defined as the number of occurrences in one entity that is associated with the number of occurrences in another entity. There are the three degrees of relationship:
- One-to-One (1:1)
- One-to-Many (1:M)
- Many-to-Many (M: M)

---
    
## Relational Data Model
## Normalisation
## Transaction Processing
## Concurrency Control
## File Organization
## Hasing
## Raid