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

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept2.png" align=center  />

In addition, an entity that depends on another entity is called a *weak entity*. The weak entity doesn't contain any key attribute of its own. The weak entity is represented by a double rectangle.

> DIFFERENT NOTATION STYLES MAY HAVE DIFFERENT REPRESENTATIONS

#### Attributes
The attribute is used to describe the property of an entity. Eclipse is used to represent an attribute.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept4.png" align=center  />

##### Key Attribute
The key attribute is used to represent the main characteristics of an entity. It represents a primary key. The key attribute is represented by an ellipse with the text underlined.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept5.png" align=center  />

##### Composite Attribute
An attribute composed of many other attributes is known as a composite attribute. The composite attribute is represented by an ellipse, and those ellipses are connected with an ellipse.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept6.png" align=center  />

##### Multivaled Attribute
An attribute can have more than one value. These attributes are known as multivalued attributes. The double oval is used to represent multivalued attributes.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept7.png" align=center  />

##### Derived Attribute
An attribute that can be derived from another attribute is known as a derived attribute. It can be represented by a dashed ellipse.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept8.png" align=center  />

---

#### Relationship
A relationship is used to describe the relation between entities. A diamond or rhombus is used to represent the relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept9.png" align=center  />

##### One to One Relationship
When only one instance of an entity is associated with the relationship, then it is known as one to one relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept10.png" align=center  />

##### One to Many Relationship
When only one instance of the entity on the left, and more than one instance of an entity on the right is associated with the relationship then this is known as a one-to-many relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept11.png" align=center  />

##### Many to One Relationship
When more than one instance of the entity on the left, and only one instance of an entity on the right are associated with the relationship then it is known as a many-to-one relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept12.png" align=center  />

##### Many to Many Relationship
When more than one instance of the entity on the left, and more than one instance of an entity on the right are associated with the relationship then it is known as a many-to-many relationship.

<img src="https://static.javatpoint.com/dbms/images/dbms-er-model-concept13.png" align=center  />

### Notation of ERD
The database can be represented using the notations. In ER diagram, many notations are used to express the cardinality. These notations are as follows:

<img src="https://static.javatpoint.com/dbms/images/dbms-notation-of-er-diagram.png" align=center  />

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

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints.png" align=center  />

##### One-to-Many
In a one-to-many mapping, an entity in E1 is associated with any number of entities in E2, and an entity in E2 is associated with at most one entity in E1.

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints2.png" align=center  />

##### Many-to-One
In a one-to-many mapping, an entity in E1 is associated with at most one entity in E2, and an entity in E2 is associated with any number of entities in E1.

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints3.png" align=center  />

##### Many-to-Many
In many-to-many mapping, an entity in E1 is associated with any number of entities in E2, and an entity in E2 is associated with any number of entities in E1.

<img src="https://static.javatpoint.com/dbms/images/dbms-mapping-constraints4.png" align=center  />

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

<img src="https://static.javatpoint.com/dbms/images/dbms-reduction-of-er-diagram-into-table.png" align=center  />

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

<img src="https://static.javatpoint.com/dbms/images/dbms-reduction-of-er-diagram-into-table2.png" align=center  />

---

#### Relationship of Higher Degree
The degree of relationship can be defined as the number of occurrences in one entity that is associated with the number of occurrences in another entity. There are the three degrees of relationship:
- One-to-One (1:1)
- One-to-Many (1:M)
- Many-to-Many (M: M)

---
    
## Relational Data Model
### Relational Model Concept
The relational model can represent as a table columns and rows. Each row is known as a tuple. Each table of the column has a name or attribute.

- **Domain**: It contains a set of atomic values that an attribute can take.
- **Attribute**: It contains the name of a column in a particular table. Each attribute $A_i$ must have a domain, $\text{dom}(A_i)$.
- **Relational Instance**: In the relational database system, the relational instance is represented by a finite set of tuples. Relation instances do not have duplicate tuples.
- **Relational Schema**: A relational schema contains the name of the relation and the name of all columns or attributes.
- **Relational Key**: In the relational key, each row has one or more attributes. It can identify the row in the relation uniquely.

#### Properties of Relations
- Name of the relation is distinct from all other relations.
- Each relation cell contains exactly one atomic (single) value
- Each attribute contains a distinct name
- Attribute domain has no significance
- Tuple has no duplicate value
- Order of tuples can have a different sequence

### Relational Algebra
Relational algebra is a procedural query language. It gives a step-by-step process to obtain the result of the query. It uses operators to perform queries.

#### Types of Relational Operation
There are seven types of relational operation: select operation, project operation, union operation, set intersection, set difference, cartesian product and rename operation. 

##### Select Operation
- The select operation selects tuples that satisfy a given predicate.
- It is denoted by sigma($\sigma$)
- $\sigma$ is used for selection prediction
- $r$ is used for relation
- $p$ is used as a propositional logical formula which may use connectors like: AND, OR, and NOT. These relational can use as relational operators like  =, ???, ???, <, >, ???.

##### Project Operation
- This operation shows the list of those attributes that we wish to appear in the result. The rest of the attributes are eliminated from the table. 
- It is denoted by ???. `??? NAME, CITY (CUSTOMER)` (project NAME, CITY from table CUSTOMER)

##### Union Operation
- Suppose there are two tuples R and S. The union operation contains all the tuples that are either in R or S or both in R & S.
- It eliminates the duplicate tuples. It is denoted by ???.
- A union operation must hold the following condition:
    - R and S must have the attribute of the same number
    - Duplicate tuples are eliminated automatically

##### Set Intersection
- Suppose there are two tuples R and S. The set intersection operation contains all tuples that are in both R & S. 
- It is denoted by intersection ???.

##### Set Difference
- Suppose there are two tuples R and S. The set intersection operation contains all tuples that are in R but not in S.
- It is denoted by intersection minus (-).

##### Cartesian Product
- The Cartesian product is used to combine each row in one table with each row in the other table. It is also known as a cross-product.
- It is denoted by X.

##### Rename Operation
- The rename operation is used to rename the output relation. It is denoted by rho (??).

### Join Operation
A join operation combines related tuples from different relations, if and only if a given join condition is satisfied. It is denoted by ???.

#### Types of Join
##### Natural Join
- A natural join is the set of tuples of all combinations in R and S that are equal on their common attribute names.
- It is denoted by ???.

##### Outer Join
- The outer join operation is an extension of the join operation. It is used to deal with missing information.

###### Left Outer Join
- Left outer join contains the set of tuples of all combinations in R and S that are equal on their common attribute names.
- In the left outer join, tuples in R have no matching tuples in S.
- It is denoted by ???.

###### Right Outer Join
- Right outer join contains the set of tuples of all combinations in R and S that are equal on their common attribute names.
- In the right outer join, tuples in S have no matching tuples in R.
- It is denoted by ???.- 

###### Full Outer Join
- Full outer join is like a left or right join except that it contains all rows from both tables.
- In full outer join, tuples in R have no matching tuples in S and tuples in S that have no matching tuples in R in their common attribute name.
- It is denoted by ???.- 

##### Equal Join
It is also known as an inner join. It is the most common join. It is based on matched data as per the equality condition. The equi join uses the comparison operator(=).

### Integrity Constraints
- Integrity constraints are a set of rules. It is used to maintain the quality of information.
- Integrity constraints ensure that the data insertion, updating, and other processes have to be performed in such a way that data integrity is not affected.
- Thus, integrity is used to guard against accidental damage to the database.

#### Domain Constraints
- Domain constraints can be defined as the definition of a valid set of values for an attribute.
- The data type of domain includes string, character, time, date, currency, etc. The value of the attribute must be available in the corresponding domain.

#### Entity Integrity Constraints
- The entity integrity constraint states that the primary key value can't be null.
- This is because the primary key value is used to identify individual rows in relation and if the primary key has a null value, then we can't identify those rows.
- A table can contain a null value other than the primary key field.

#### Referential Integrity Constraints
- A referential integrity constraint is specified between two tables.
- In the Referential integrity constraints, if a foreign key in Table 1 refers to the Primary Key of Table 2, then every value of the Foreign Key in Table 1 must be null or be available in Table 2.

#### Key Constraints
- Keys are the entity set that is used to identify an entity within its entity set uniquely.
- An entity set can have multiple keys, but out of which one key will be the primary key. A primary key can contain a unique and null value in the relational table.

### Relational Calculus
There is an alternate way of formulating queries known as Relational Calculus. Relational calculus is a non-procedural query language. In the non-procedural query language, the user is concerned with the details of how to obtain the results. The relational calculus tells what to do but never explains how to do it. Most commercial relational languages are based on aspects of relational calculus including SQL-QBE and QUEL.

It is based on Predicate calculus, a name derived from a branch of symbolic language. A predicate is a truth-valued function with arguments. On substituting values for the arguments, the function result in an expression called a proposition. It can be either true or false. It is a tailored version of a subset of the Predicate Calculus to communicate with the relational database.

Many calculus expressions involve the use of Quantifiers. There are two types of quantities:
- Universal Quantifiers: The universal quantifier denoted by ??? is read as for all which means that in a given set of tuples exactly all tuples satisfy a given condition.
- Existential Quantifiers: The existential quantifier denoted by ??? is read as for all which means that in a given set of tuples there is at least one occurrence whose value satisfies a given condition.

Before using the concept of quantifier in formulas, we need to know the concept of Free and Bound Variables.

A tuple variable t is bound if it is quantified which means that if it appears in any occurrences a variable that is not bound is said to be free.

Free and bound variables may be compared with the global and local variables of programming languages.

#### Tuple Relational Calculus (TRC)
It is a non-procedural query language that is based on finding several tuple variables also known as range variables for which the predicate holds. It describes the desired information without giving a specific procedure for obtaining that information. The tuple relational calculus is specified to select the tuples in a relation. In TRC, the filtering variable uses the tuples of a relation. The result of the relation can have one or more tuples.

#### Domain Relational Calculus (DRC)
The second form of relation is known as Domain relational calculus. In domain relational calculus, filtering variable uses the domain of attributes. Domain relational calculus uses the same operators as tuple calculus. It uses logical connectives ??? (and), ??? (or) and ??? (not). It uses Existential (???) and Universal Quantifiers (???) to bind the variable. The QBE or Query by example is a query language related to domain relational calculus.

---

## Normalisation
- Normalization is the process of organizing the data in the database.
- Normalization is used to minimize the redundancy from a relation or set of relations. It is also - used to eliminate undesirable characteristics like Insertion, Update, and Deletion Anomalies.
- Normalization divides the larger table into smaller and links them using relationships.
- The normal form is used to reduce redundancy from the database table.

The main reason for normalizing the relations is to remove these anomalies. Failure to eliminate anomalies leads to data redundancy and can cause data integrity and other problems as the database grows. Normalization consists of a series of guidelines that helps to guide you in creating a good database structure.

### Function Dependency
Functional dependency is a relationship that exists between two attributes. It typically exists between the primary key and non-key attribute within a table. The left side of FD is known as a determinant, and the right side of the production is known as a dependent.

#### Types of Functional Dependency
There are two types of functional dependency, trivial and non-trivial functional dependency.

##### Trivial Functional Dependency
- A ??? B has trivial functional dependency if B is a subset of A.
- The following dependencies are also trivial: A ??? A, B ??? B.

```
Example
Consider a table with two columns Employee_Id and Employee_Name.
{Employee_id, Employee_Name} ??? Employee_Id is a trivial functional dependency 
                    as Employee_Id is a subset of {Employee_Id, Employee_Name}.  
Also, Employee_Id ??? Employee_Id and Employee_Name ??? Employee_Name 
                    are trivial dependencies too.  
```

##### Non-trivial Functional Dependency
- A ??? B has a non-trivial functional dependency if B is not a subset of A.
- When A intersection B is NULL, then A ??? B is called a complete non-trivial.

### Inference Rule
- Armstrong's axioms are the basic inference rule.
- Armstrong's axioms are used to conclude functional dependencies on a relational database.
- The inference rule is a type of assertion. It can apply to a set of FD (functional dependency) to derive other FD.
- Using the inference rule, we can derive additional functional dependency from the initial set.

The function dependency has 6 types of inference rules.

#### Reflexive Rule ($IR_1$)
In the reflexive rule, if Y is a subset of X, then X determines Y.

If X ??? Y then X ??? Y.

#### Augmentation Rule ($IR_2$)
The augmentation is also called a partial dependency. In augmentation, if X determines Y, then XZ determines YZ for any Z.

If X ??? Y then XZ ??? YZ.

#### Transitive Rule ($IR_3$)
In the transitive rule, if X determines Y and Y determines Z, then X must also determine Z.

If X ??? Y and Y ??? Z then X ??? Z. 

#### Union Rule ($IR_4$)
Union rule says, if X determines Y and X determines Z, then X must also determine Y and Z.

If X ??? Y and X ??? Z then X ??? YZ.

#### Decomposition Rule ($IR_5$)
The decomposition rule is also known as the project rule. It is the reverse of the union rule.

This rule says, if X determines Y and Z, then X determines Y and X determines Z separately.

If X ??? YZ then X ??? Y and X ??? Z.

#### Pseudo Transitive Rule ($IR_6$)
In Pseudo transitive Rule, if X determines Y and YZ determines W, then XZ determines W.

If X ??? Y and YZ ??? W then XZ ??? W.

### DBMS Normalisation
A large database defined as a single relation may result in data duplication. This repetition of data may result in:
- Marketing relations are very large.
- It isn't easy to maintain and update data as it would involve searching many records in relation.
- Wastage and poor utilization of disk space and resources.
- The likelihood of errors and inconsistencies increases.

So to handle these problems, we should analyze the decompose the relations with redundant data into smaller, simpler, and well-structured relations that satisfy desirable properties. Normalisation is a process of decomposing the relations into relations with fewer attributes.

Data modification anomalies can be categorized into three types:
- **Insertion Anomaly**: Insertion Anomaly refers to when one cannot insert a new tuple into a relationship due to a lack of data.
- **Deletion Anomaly**: The delete anomaly refers to the situation where the deletion of data results in the unintended loss of some other important data.
- **Updatation Anomaly**: The update anomaly is when an update of a single data value requires multiple rows of data to be updated.

#### Types of Normal Forms
Normalisation works through a series of stages called Normal forms. The normal forms apply to individual relations. The relation is said to be in particular normal form if it satisfies constraints.

![](https://static.javatpoint.com/dbms/images/dbms-normalization.png)

| Normal Form | Description |
| ---- | ---- |
| 1NF | A relation is in 1NF if it contains an atomic value |
| 2NF | A relation will be in 2NF if it is in 1NF and all non-key attributes are fully functional and dependent on the primary key |
| 3NF | A relation will be in 3NF if it is in 2NF and no transition dependency form |
| BCNF | A stronger definition of 3NF is known as Boyce Codd's normal form |
| 4NF | A relation will be in 4NF if it is in Boyce Codd's normal form and has no multi-valued dependency |
| 5NF | A relation is in 5NF. If it is in 4NF and does not contain any join dependency, joining should be lossless |

#### DBMS 1NF
- A relation will be 1NF if it contains an atomic value.
- It states that an attribute of a table cannot hold multiple values. It must hold only a single-valued attribute.
- First normal form disallows the multi-valued attribute, composite attribute, and their combinations.

Example: Relation EMPLOYEE is not in 1NF because of the multi-valued attribute EMP_PHONE.

| EMP_ID | EMP_NAME | EMP_PHONE | EMP_STAT | 
| ---- | ---- | ---- | ---- |
| 14 | John | 7272826385<br>9064738238 | UP |
| 20 | Harry | 8574783832 | Biha | 
| 12 | Sam | 7390372389<br>8589830302 | Punja | 

The decomposition of the EMPLOYEE table into 1NF has been shown below:

| EMP_ID | EMP_NAME | EMP_PHONE | EMP_STAT |
| ---- | ---- | ---- | ---- |
| 14 | John | 7272826385 | UP |
| 14 | John | 9064738238 | UP |
| 20 | Harry | 8574783832 | Biha | 
| 12 | Sam | 7390372389 | Punja | 
| 12 | Sam | 8589830302 | Punja | 

#### DBMS 2NF
- In the 2NF, relational must be in 1NF.
- In the second normal form, all non-key attributes are fully functional and dependent on the primary key.

Example: Let's assume, a school can store the data of teachers and the subjects they teach. In a school, a teacher can teach more than one subject.

| TEACHER_ID | SUBJECT | TEACHER_AGE|
| ---- | ---- | ---- | 
| 25 | Chemistry | 30| 
| 25 | Biology | 30| 
| 47 | English | 35| 
| 83 | Math | 38| 
| 83 | Computer | 38| 

In the given table, the non-prime attribute TEACHER_AGE is dependent on TEACHER_ID which is a proper subset of a candidate key. That's why it violates the rule for 2NF.

To convert the given table into 2NF, we decompose it into two tables:

| TEACHER_ID | TEACHER_AGE | 
| ---------- | ----------- |
| 25         | 30          | 
| 47         | 35          | 
| 83         | 38          | 

| TEACHER_ID | SUBJECT   |
| ---------- | --------- |
| 25         | Chemistry | 
| 25         | Biology   | 
| 47         | English   | 
| 83         | Math      | 
| 83         | Computer  | 

#### DBMS 3NF
- A relation will be in 3NF if it is in 2NF and does not contain any transitive partial dependency.
- 3NF is used to reduce data duplication. It is also used to achieve data integrity.
- If there is no transitive dependency for non-prime attributes, then the relationship must be in the third normal form.

A relation is in third normal form if it holds at least one of the following conditions for every non-trivial function dependency X ??? Y.
1. X is a super key.
2. Y is a prime attribute, i.e., each element of Y is part of some candidate key.

| EMP_ID | EMP_NAME | EMP_ZIP | EMP_STATE | EMP_CITY |
| ---- | ---- | ---- | ---- | ---- |
| 222 | Harry | 201010 | UP | Noida |
| 333 | Stephan | 02228 | US | Boston |
| 444 | Lan	60007 | US | Chicago |
| 555 | Katharine | 06389 | UK	Norwich |
| 666 | John | 462007 | MP | Bhopal |

Super key in the table above:
- {EMP_ID}, {EMP_ID, EMP_NAME}, {EMP_ID, EMP_NAME, EMP_ZIP}....so on  

Candidate key: {EMP_ID}

Non-prime attributes: In the given table, all attributes except EMP_ID are non-prime.

Here, EMP_STATE & EMP_CITY dependent on EMP_ZIP and EMP_ZIP dependent on EMP_ID. The non-prime attributes (EMP_STATE, EMP_CITY) are transitively dependent on the super key(EMP_ID). It violates the rule of the third normal form.

That's why we need to move the EMP_CITY and EMP_STATE to the new <EMPLOYEE_ZIP> table, with EMP_ZIP as a Primary key.

| EMP_ID | EMP_NAME | EMP_ZIP | 
| ---- | ---- | ---- |
| 222 | Harry | 201010 | 
| 333 | Stephan | 02228 | 
| 444 | Lan | 60007 | 
| 555 | Katharine | 06389 | 
| 666 | John | 462007 | 

| EMP_ZIP | EMP_STATE | EMP_CITY |
| ---- | ---- | ---- |
| 201010 | UP | Noida | 
| 02228 | US | Boston | 
| 60007 | US | Chicago | 
| 06389 | UK | Norwich | 
| 462007 | MP | Bhopal | 

#### DBMS BCNF
- BCNF is the advanced version of 3NF. It is stricter than 3NF.
- A table is in BCNF if every functional dependency X ??? Y, X is the super key of the table.
- For BCNF, the table should be in 3NF, and for every FD, LHS is super key.

Example: Let's assume there is a company where employees work in more than one department.

| EMP_ID | EMP_COUNTRY | EMP_DEPT | DEPT_TYPE | EMP_DEPT_NO |
| ---- | ---- | ---- | ---- | ---- |
| 264 | India | Designing | D394 | 283 |
| 264 | India | Testing | D394 | 300 |
| 364 | UK | Stores | D283 | 232 |
| 364 | UK | Developing | D283 | 549 |

In the above table Functional dependencies are as follows:
- EMP_ID  ???  EMP_COUNTRY  
- EMP_DEPT  ???   {DEPT_TYPE, EMP_DEPT_NO}  

Candidate key: {EMP-ID, EMP-DEPT}

The table is not in BCNF because neither EMP_DEPT nor EMP_ID alone are keys.

To convert the given table into BCNF, we decompose it into three tables:

| EMP_ID | EMP_COUNTRY | 
| ---- | ---- |
| 264 | India | 
| 264 | India | 

| EMP_DEPT | DEPT_TYPE | EMP_DEPT_NO | 
| ---- | ---- | ---- |
| Designing | D394 | 283 | 
| Testing | D394 | 300 | 
| Stores | D283 | 232 | 
| Developing | D283 | 549 | 

| EMP_ID | EMP_DEPT | 
| ---- | ---- |
| D394 | 283 | 
| D394 | 300 | 
| D283 | 232 | 
| D283 | 549 | 

Functional dependencies:
- EMP_ID ??? EMP_COUNTRY  
- EMP_DEPT ??? {DEPT_TYPE, EMP_DEPT_NO}  

Candidate keys:
- For the first table: EMP_ID
- For the second table: EMP_DEPT
- For the third table: {EMP_ID, EMP_DEPT}

Now, this is in BCNF because the left side part of both the functional dependencies is a key.

#### DBMS 4NF
- A relation will be in 4NF if it is in Boyce Codd's normal form and has no multi-valued dependency.
- For a dependency A ??? B, if for a single value of A, multiple values of B exist, then the relationship will be a multi-valued dependency.

| STU_ID | COURSE | HOBBY |
| ---- | ---- | ---- |
| 21 | Computer | Dancing | 
| 21 | Math | Singing | 
| 34 | Chemistry | Dancing | 
| 74 | Biology | Cricket | 
| 59 | Physics | Hockey | 

The given STUDENT table is in 3NF, but the COURSE and HOBBY are two independent entities. Hence, there is no relationship between COURSE and HOBBY.

In the STUDENT relation, a student with STU_ID, 21 contains two courses, Computer and Math and two hobbies, Dancing and Singing. So there is a Multi-valued dependency on STU_ID, which leads to unnecessary repetition of data.

So to make the above table into 4NF, we can decompose it into two tables:

| STU_ID | COURSE | 
| ---- | ---- |
| 21 | Computer | 
| 21 | Math | 
| 34 | Chemistry | 
| 74 | Biology | 
| 59 | Physics | 

| STU_ID | HOBBY |
| ---- | ---- |
| 21 | Dancing | 
| 21 | Singing | 
| 34 | Dancing | 
| 74 | Cricket | 
| 59 | Hockey |  

#### DBMS 5NF
- A relation is in 5NF if it is in 4NF and does not contain any join dependency and joining should be lossless.
- 5NF is satisfied when all the tables are broken into as many tables as possible to avoid redundancy.
- 5NF is also known as Project-join normal form (PJ/NF).

| SUBJECT | LECTURER | SEMESTER | 
| ---- | ---- | ---- |
| Computer | Anshika | Semester 1 | 
| Computer | John | Semester 1 | 
| Math | John | Semester 1 | 
| Math | Akash | Semester 2 | 
| Chemistry | Praveen | Semester 1 | 

In the above table, John takes both Computer and Math classes for Semester 1 but he doesn't take Math classes for Semester 2. In this case, a combination of all these fields is required to identify valid data.

Suppose we add a new Semester as Semester 3 but do not know about the subject and who will be taking that subject so we leave Lecturer and Subject as NULL. But all three columns together act as a primary key, so we can't leave the other two columns blank.

So to make the above table into 5NF, we can decompose it into three relations P1, P2 & P3:

| SEMESTER | SUBJECT | 
| ---- | ---- |
| Semester 1 | Computer | 
| Semester 1 | Math | 
| Semester 1 | Chemistry | 
| Semester 2 | Math | 

| SUBJECT | LECTURER | 
| ---- | ---- |
| Computer | Anshika | 
| Computer | John | 
| Math | John | 
| Math | Akash | 
| Chemistry | Praveen | 

| SEMESTER | LECTURER | 
| ---- | ---- |
| Semester 1 | Anshika | 
| Semester 1 | John | 
| Semester 1 | John | 
| Semester 2 | Akash | 
| Semester 1 | Praveen | 

#### Advantages vs. Disadvantages of Normalisation
| Advantages | Disadvantages |
| ---------- | ------------- |
| Normalisation helps to minimize data redundancy | You cannot start building the database before knowing what the user needs |
| Greater overall database organisation | The performance degrades when normalizing the relations to higher normal forms, i.e., 4NF, 5NF |
| Data consistency within the database | It is very time-consuming and difficult to normalize relations of a higher degree |
| Much more flexible database design | Careless decomposition may lead to a bad database design, leading to serious problems | 
| Enforces the concept of relational integrity | |

### Relational Decomposition
- When a relation in the relational model is not inappropriate normal form then the decomposition of a relationship is required.
- In a database, it breaks the table into multiple tables.
- If the relation has no proper decomposition, then it may lead to problems like loss of information.
- Decomposition is used to eliminate some of the problems of bad design like anomalies, inconsistencies, and redundancy.

#### Lossless Decomposition
- If the information is not lost from the relation that is decomposed, then the decomposition will be lossless.
- The lossless decomposition guarantees that the join of relations will result in the same relation as it was decomposed.
- The relation is said to be lossless decomposition if natural joins of all the decomposition give the original relation.

#### Dependency Preserving
- It is an important constraint of the database.
- In the dependency preservation, at least one decomposed table must satisfy every dependency.
- If a relation R is decomposed into relation R1 and R2, then the dependencies of R either must be a part of R1 or R2 or must be derivable from the combination of functional dependencies of R1 and R2.
- For example, suppose there is a relation R (A, B, C, D) with a functional dependency set (A->BC). The relational R is decomposed into R1(ABC) and R2(AD) which is dependency preserving because FD A->BC is a part of relation R1(ABC).

### Multivalued Dependency
- Multivalued dependency occurs when two attributes in a table are independent of each other but, both depend on a third attribute.
- A multivalued dependency consists of at least two attributes that are dependent on a third attribute that's why it always requires at least three attributes.

### Join Dependency
- Join decomposition is a further generalization of Multivalued dependencies.
- If the join of R1 and R2 over C is equal to relation R, then we can say that a join dependency (JD) exists.
- Where R1 and R2 are the decompositions R1(A, B, C) and R2(C, D) of a given relations R (A, B, C, D).
- Alternatively, R1 and R2 are a lossless decomposition of R.
- A JD ??? {R1, R2,..., Rn} is said to hold over a relation R if R1, R2,....., Rn is a lossless-join decomposition.
- The \*(A, B, C, D), (C, D) will be a JD of R if the join of join's attribute is equal to the relation R.
- Here, \*(R1, R2, R3) is used to indicate that relation R1, R2, R3 and so on are a JD of R.

### Inclusion Dependence
- Multivalued dependency and join dependency can be used to guide database design although they both are less common than functional dependencies.
- Inclusion dependencies are quite common. They typically show little influence on the design of the database.
- The inclusion dependency is a statement in which some columns of a relation are contained in other columns.
- The example of inclusion dependency is a foreign key. In one relation, the referring relation is contained in the primary key column(s) of the referenced relation.
- Suppose we have two relations R and S which were obtained by translating two entity sets such that every R entity is also an S entity.
- Inclusion dependency would happen if projecting R on its key attributes yields a relation that is contained in the relation obtained by projecting S on its key attributes.
- In inclusion dependency, we should not split groups of attributes that participate in an inclusion dependency.
- In practice, most inclusion dependencies are key-based that is involved only keys.

### Canonical Cover
In the case of updating the database, the responsibility of the system is to check whether the existing functional dependencies are getting violated during the process of updating. In case of a violation of functional dependencies in the new database state, the rollback of the system must take place.

A canonical cover or irreducible set of functional dependencies FD is a simplified set of FD that has a similar closure as the original set FD.

An attribute of an FD is said to be extraneous if we can remove it without changing the closure of the set of FD.

---

## Transaction Processing
### Transaction
- The transaction is a set of logically related operation. It contains a group of tasks.
- A transaction is an action or serires of actions. It is performed by a single user to perform operations for accessing the contents of the database.

**Operations of Transaction**

Following are the main operations of transaction.
- READ(X): Read operation is used to read the value of X from the database and stores it in a buffer in main memory.
- WRITE(X): Write operation is used to write the value back to the database from the buffer.

### Transaction Property
The transaction has four properties. These are used to maintain consistency in a database, before and after the transaction.

#### Property of Transaction
##### Atomoicity
- It states that all operations of the transaction take place at once if not, the transaction is aborted.
- There is no widway, i.e., the transaction cannot occur partially. Each transaction is treated as one unit and either run to copmletion or is not executed at all.

Atommicity involves the following two operations:
- Abort: If a transaction aborts then all the changes made are not visible.
- Commit: If a transaction commits then all the changes made are visible.

##### Consistency 
- The integrity constraints are maintained so that the database is consistent before and after the transaction.
- The execution of a transaction will leave a database in either its prior stable state or a new stable state.
- The consistent property of database states that every transaction sees a consistent database instance.
- The transaction is used to transform the database from one consistent state to another consistent state.

##### Isolation
- It shows that the data which is used at the time of execution of a transaction cannot be used by the second transaction until the first one is completed.
- In isolation, if the transaction T1 is being executed and using the data item X, then that data item can't be accessed by any other transaction T2 until the transaction T1 ends.
- The concurrency control subsystem of the DBMS enforced the isolation property.

##### Durability
- The durability property is used to indicate the performance of the database's consistent state. It states that the transaction made the permanent changes.
- They cannot be lost by the erroneous operation of a faulty transaction or by the system failure. When a transaction is completed, then the database reaches a state known as the consistent state. That consistent state cannot be lost, even in the event of a system's failure.
- The recovery subsystem of the DBMS has the responsibility of Durability property.

### States of Transaction

![](https://static.javatpoint.com/dbms/images/dbms-states-of-transaction.png)

In a database, the transaction can be in one of the following states:
- Active State
    - The active state is the first state of every transaction. In this state, the transaction is being executed.
- Partialy commited
    - In the partially commited state, a transaction executes its final operation, but the data is still not saved to the database.
- Commited
    - A transaction is said to be in a committed state if it executes all its operations successfully. In this state, all the effects are now permanently saved on the database system.
- Failed state
    - If any of the checks made by the database recovery system fails, then the transaction is said to be in the failed state.
- Aborted
    - If any of the checks fail and the transaction has reached a failed state then the database recovery system will make sure that the database is in its previous consistent state. If not then it will abort or roll back the transaction to bring the database in a consistency state.
    - If the transaction fails in the middle of the transaction then before executing the transaction, all executed transactions are rolled back to its consistent state.
    - After aborting the transaction, the database recovery module will select one of the two operations:
        - Re-start the transaction
        - Kill the transaction

### DBMS Schedule
A series of operation from one transaction to another transaction is known as schedule. It is used to preserve the order of the operation in each of the individual transaction.

#### Serial Schedule
The serial schedule is a type of schedule where one transaction is executed completely before starting another transaction. In the serial schedule, when the first transaction completes its cycle, then the next transaction is executed.

#### Non-Serial Schedule
If interleaving of operations is allowed, then there will be non-serial schedule. It contains many possible orders in which the system can execute the individual operations of the transactions.

#### Serializable Schedule
The serializability of schedules is used to find non-serial schedules that allow the transaction to execute concurrently without interfering with one another. It identifies which schedules are correct when executions of the transaction have interleaving of their operations. A non-serial schedule will be serializable if its result is equal to the result of its transactions executed serially.

### Conflict Schedule
A schedule is called conflict serializability if after swapping of non-conflicting operations, it can transform into a serial schedule.
The schedule will be a conflict serializable if it is conflict equivalent to a serial schedule.

#### Conflict Operations
The two operations become conflicting if all conditions satisfy:
- Both belong to seperate transcations.
- They have the same data item.
- They contain at least one write operation.

#### Conflict Equivalent
In the conflict equivalent, one can be transformed to another by swapping non-conflicting operations. Two schedules are said to be conflict equivalent if and only if:
- They contain the same set of the transaction.
- If each pair of conflict operations are ordered in the same way.

### View Serializability
- A schedule will view serializable if it is view equivalent to a serial schedule.
- If a schedule is conflict serializable, then it will be view serializable.
- The view serializable which does not conflict serializable contains blind writers.

### Recoverability of Schedule
Sometimes a transaction may not execute completely due to a software issue, system crash or hardware failure. In that case, the failed transation has to be rollback. But some other transaction may also have used value produced by the failed transaction. So we have to rollback those transactions.

> **Irrecoverable schedule**: The schedule will be irrecoverable if T_j reads the updated value of T_i and T_j commited before T_i commit.

> **Recoverable with cascading rollback**: The schedule will be recoverable with cascading rollback if T_j reads the updated value of T_i. Commit of T_j is delayed till commit of T_i.

### Failure Classification
To find that where the problem has occurred, we generalize a failure into the following categories:
- The transaction failure occurs if fails to execute or when it reaches a point from where it can't go any further. If a few transaction or process is hurt, then this is called as transaction failure. Reasons for a transaction failure could be:
    - Logical errors: If a transaction cannot complete due to some code error or an internal error condition, then the logical error occurs.
    - Syntax errors: If occurs where the DBMS itself terminates an active transation because the database system is not able to execute it. 
- System crash failure can occur due to power failure or other hardware or software failure.
- Disk Failure
    - It occurs where hard-disk drives or storage drives used to fail frequently. It was a common problem in the early days of technology evolution.
    - Disk failure occurs due to the formation of bad sectors, disk head crash, and unreachability to the disk or any other failure, which destroy all or part of disk storage.

### Log-Based Recovery
- The log is a sequence of records. Log of each transaction is maintained in some stable storage so that if any failure occurs, then it can be recovered from there.
- If any operation is performed on the database, then it will be recorded in the log.
- But the process of storing the logs should be done before the actual transaction is applied in the database.

- Deferred database modification
    - The deferred modification technique occurs if the transation does not modify the database until it has commited.
    - In this method, all the logs are created and stored in the stable storage, and the database is updated when a transaction commits.
- Immediate database modification
    - The immediate modification technique occurs if database modification occurs while the transaction is still active.
    - In this technique, the database is modified immeidately after every operation. It follows an actual database modification.

**Recovery using Log records**
When the system is crashed, then the system consults the log to find which transactions need to be undone and which need to be redone.
- If the log contains the record <T_i, start> and <T_i, commit>, then the transation T_i needs to be redone.
- If log contains record <T_n, start> but does not contain the record either <T_i, commit> or <T_i, abort>, then the Transaction T_i to be undone.

### DBMS Checkpoint
- The checkpoint is a type of mechansim where all the previous logs are removed from the system and permanelty stored in the storage disk.
- The checkpoint is like a bookmart. While the execution of the transation, such checkpoints are marked, and the transaction is executed then using the steps of the transation, the log files will be created.
- When it reaches to the checkpoint, then the transaction will be updated into the database, and till that point, the entire log file will be removed from the file. Then the log file is updated with the new step of transaction till next checkpoint and so on.
- The checkpoint is used to declare a point before which the DBMS was in the consistent state, and all transaction were commited.

### Deadlock in DBMS
A deadlock is a condition where two or more transaction are waiting indefinitely for one another to give up locks. Deadlocks is said to be one of the most feared complications in DBMS as no task ever gets finished and is in waiting state forever.

**Deadlock Avoidance**
- When a database is stuck in a deadlock state, then it is better to avoid the database rather than aborting or restating the database. This is a waste of time and resource.
- Deadlock avoidance mechanism is used to detect any deadlock situation in advance. A method like "wait for graph" is used for detecting the deadlock situation but this method is suitable only for the smaller database. For the larger database, deadlock prevention method can be used.

**Deadlock Detection**
In a database, when a transaction waits indefinitely to obtain a lock, then the DBMS should detect whether the transaction is involved in a deadlock or not. The lock manager maintains a Wait for the graph to detect the deadlock cycle in the database.

**Deadlock Prevention**
- Deadlock prevention method is suitable for a large database. If the resources are allocated in such a way that deadlock never occurs, then the deadlock can be prevented.
- The Database management system analyzes the operations of the transaction whether they can create a deadlock situation or not. If they do, then the DBMS never allowed that transaction to be executed.

---

## Concurrency Control
### Concurrency Control

### Lock-Based Protocol 

### Time Stamp Protocol

### Validation-Based Protocol

### Thomas Write Rule

### Multiple Granularity

### Recorvery Concurrency Transaction

---

## File Organization

### File organisation 

### Sequential File Organisation

### Heap File Organisation

### Hash File Organisation

### B+ File Organisation

### Cluster File Organisation

---

## Indexing and B+ Tree
### Indexing in DBMS

### B+ Tree

---

## Hashing

### Hashing

### Static Hashing

### Dynamic Hashing

---

## Raid
