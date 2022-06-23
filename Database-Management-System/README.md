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
Relational model can represent as a table columns and rows. Each row is known as a tuple. Each table of the column has a name or attribute.

- **Domain**: It contains a set of atomic values that an attribute can take.
- **Attribute**: It contains the name of a column in a particular table. Each attribute $A_i$ must have a domain, $\text{dom}(A_i)$.
- **Relational Instance**: In the relational database system, the relational instance is represented by finite set of tuples. Relation instances do not have duplicate duples.
- **Relational Schema**: A relational schema contains the name of the relation and name of all columns or attribtues.
- **Relational Key**: In the relational key, each row has one or more attributes. It can identify the row in the relation uniquely.

#### Properties of Relations
- Name of the relation is distinct from all other relations.
- Each relation cell contains exactly one atomic (single) value
- Each attribute contains a distinct name
- Attribute domain has no significance
- Tuple has no duplicate value
- Order of tuple cn have a different sequence

### Relational Algebra
Relational algebra is a procedural query language. It gives a step by step process to obtain result of the query. It uses operators to perform queries.

#### Types of Relational Operation
There are seven types of relational operation: select operation, project operation, union operation, set intersetion, set difference, cartesian product and rename operation. 

##### Select Operation
- The select operation selects tuples that satisfy a given predicate.
- It is denoted by sigma($\sigma$)
- $\sigma$ is used for selection prediction
- $r$ is used for relation
- $p$ is used as a propositional logical formula which may use connectros like: AND, OR, and NOT. These relational can use as relational operators like  =, ≠, ≥, <, >, ≤.

##### Project Operation
- This operation shows the list of those attributes that we wish to appear in the result. Rest of the attributes are eliminated from the table. 
- It is denoted by ∏. `∏ NAME, CITY (CUSTOMER)` (project NAME, CITY from table CUSTOMER)

##### Union Operation
- Suppose there are two tuples R and S. The union operation contains all the tuples that are either in R or S or both in R & S.
- It eliminates the duplicate tuples. It is denoted by ∪.
- A union operation must hold the following condition:
    - R and S must have the attribute of the same number
    - Duplicate tuples are eliminated automatically

##### Set Intersection
- Suppose there are two tuples R and S. The set intersection operation contains all tuples that are in both R & S. 
- It denoted by intersection ∩.

##### Set Difference
- Suppose there are two tuples R and S. The set intersection operation contains all tuples that are in R but not in S.
- It is denoted by intersection minus (-).

##### Cartesian Product
- The Cartesian product is used to combine each row in one table with each row in the other table. It is also known as a cross product.
- It is denoted by X.

##### Rename Operation
- The rename operation is used to rename the output relation. It is denoted by rho (ρ).

### Join Operation
A join operation combines related tuples from different relations, if and only if a given join condition is satisfied. It is denoted by ⋈.

#### Types of Join
##### Natural Join
- A natural join is the set of tuples of all combinations in R and S that are equal on their common attribute names.
- It is denoted by ⋈.

##### Outer Join
- The outer join operation is an extension of the join operation. It is used to deal with missing information.

###### Left Outer Join
- Left outer joing contains the set of tuples of all combinations in R and S that are equal on their common attribute names.
- In the left outer join, tuples in R have no matrching tuples in S.
- It is denoted by ⟕.

###### Right Outer Join
- Right outer join contains the set of tuples of all combinations in R and S that are equal on their common attribute names.
- In right outer join, tuples in S have no matching tuples in R.
- It is denoted by ⟖.- 

###### Full Outer Join
- Full outer join is like a left or right join except that it contains all rows from both tables.
- In full outer join, tuples in R that have no matching tuples in S and tuples in S that have no matching tuples in R in their common attribute name.
- It is denoted by ⟗.- 

##### Equal Join
It is also known as an inner join. It is the most common join. It is based on matched data as per the equality condition. The equi join uses the comparison operator(=).

### Integrity Constraints
- Integrity constarints are a set of rules. It is used to maintain the quality of information.
- Integrity constraints ensure that the data insertion, updating, and other processes have to be performed in such a way that data integrity is not affected.
- Thus, integrity is used to guard against accidental damage to the database.

#### Domain Constraints
- Domain constraints can be defined as the definition of a valid set of values for an attribute.
- The data type of domain includes string, character, time, date, currency, etc. The value of the attribute must be available in the corresponding domain.

#### Entity Integrity Constraints
- The entity integrity constraint states that primary key value can't be null.
- This is because the primary key value is used to identify individual rows in relation and if the primary key has a null value, then we can't identify those rows.
- A table can contain a null value other than the primary key field.

#### Referential Integrity Constraints
- A referential integrity constraint is specified between two tables.
- In the Referential integrity constraints, if a foreign key in Table 1 refers to the Primary Key of Table 2, then every value of the Foreign Key in Table 1 must be null or be available in Table 2.

#### Key Constraints
- Keys are the entity set that is used to identify an entity within its entity set uniquely.
- An entity set can have multiple keys, but out of which one key will be the primary key. A primary key can contain a unique and null value in the relational table.

### Relational Calculus
There is an alternate way of formulating queries known as Relational Calculus. Relational calculus is a non-procedural query language. In the non-procedural query language, the user is concerned with the details of how to obtain the end results. The relational calculus tells what to do but never explains how to do. Most commercial relational languages are based on aspects of relational calculus including SQL-QBE and QUEL.

It is based on Predicate calculus, a name derived from branch of symbolic language. A predicate is a truth-valued function with arguments. On substituting values for the arguments, the function result in an expression called a proposition. It can be either true or false. It is a tailored version of a subset of the Predicate Calculus to communicate with the relational database.

Many of the calculus expressions involves the use of Quantifiers. There are two types of quantitiers:
- Universal Quantifiers: The universal quantifier denoted by ∀ is read as for all which means that in a given set of tuples exactly all tuples satisfy a given condition.
- Existential Quantifiers: The existential quantifier denoted by ∃ is read as for all which means that in a given set of tuples there is at least one occurrences whose value satisfy a given condition.

Before using the concept of quantifier in formulas, we need to know the concept of Free and Bound Variables.

A tuple variable t is bound if it is quantified which means that if it appears in any occurrences a variable that is not bound is said to be free.

Free and bound variable may be compared with global and local variable of programming languages.

#### Tuple Relational Calculus (TRC)
It is a non-procedural query language which is based on finding a number of tuple variables also known as range variable for which predicate holds true. It describes the desired information without giving a specific procedure for obtaining that information. The tuple relational calculus is specified to select the tuples in a relation. In TRC, filtering variable uses the tuples of a relation. The result of the relation can have one or more tuples.

#### Domain Relational Calculus (DRC)
The second form of relation is known as Domain relational calculus. In domain relational calculus, filtering variable uses the domain of attributes. Domain relational calculus uses the same operators as tuple calculus. It uses logical connectives ∧ (and), ∨ (or) and ┓ (not). It uses Existential (∃) and Universal Quantifiers (∀) to bind the variable. The QBE or Query by example is a query language related to domain relational calculus.

---

## Normalisation
- Normalization is the process of organizing the data in the database.
- Normalization is used to minimize the redundancy from a relation or set of relations. It is also - used to eliminate undesirable characteristics like Insertion, Update, and Deletion Anomalies.
- Normalization divides the larger table into smaller and links them using relationships.
- The normal form is used to reduce redundancy from the database table.

The main reason for normalizing the relations is removing these anomalies. Failure to eliminate anomalies leads to data redundancy and can cause data integrity and other problems as the database grows. Normalization consists of a series of guidelines that helps to guide you in creating a good database structure.

### Function Dependency
The functional dependency is a relationship that exists between two attributes. It typically exists between the primary key and non-key attribute within a table. The left side of FD is known as a deterimnant, the right side of the production is known as a dependent.

#### Types of Functional Dependency
There are two types of functional dependency, trivial or non-trivial functional dependency.

##### Trivial Functional Dependency
- A → B has trivial functional dependency if B is a subset of A.
- The following dependencies are also trivial like: A → A, B → B.

```
Example
Consider a table with two columns Employee_Id and Employee_Name.
{Employee_id, Employee_Name} → Employee_Id is a trivial functional dependency 
                    as Employee_Id is a subset of {Employee_Id, Employee_Name}.  
Also, Employee_Id → Employee_Id and Employee_Name → Employee_Name 
                    are trivial dependencies too.  
```

##### Non-trivial Functional Dependency
- A → B has a non-trivial functional dependency if B is not a subset of A.
- When A intersection B is NULL, then A → B is called as complete non-trivial.

### Inference Rule
- The Armstrong's axioms are the basic inference rule.
- Armstrong's axioms are used to conclude functional dependencies on a relational database.
- The inference rule is a type of assertion. It can apply to set of FD (functional dependency) to derive other FD.
- Using the inference rule, we can derive addtional functional dependency from the initial set.

The function dependency has 6 types of inference rule.

#### Reflexive Rule ($IR_1$)
In the reflexive rule, if Y is a subset of X, then X determines Y.

If X ⊇ Y then X → Y.

#### Augmentation Rule ($IR_2$)
The augmentation is also called as a partial dependency. In augmentation, if X determines Y, then XZ determines YZ for any Z.

If X → Y then XZ → YZ.

#### Transitive Rule ($IR_3$)
In the transitive rule, if X determines Y and Y determine Z, then X must also determine Z.

If X → Y and Y → Z then X → Z. 

#### Union Rule ($IR_4$)
Union rule says, if X determines Y and X determines Z, then X must also determine Y and Z.

If X → Y and X → Z then X → YZ.

#### Decomposition Rule ($IR_5$)
Decomposition rule is also known as project rule. It is the reverse of union rule.

This Rule says, if X determines Y and Z, then X determines Y and X determines Z separately.

If X → YZ then X → Y and X → Z.

#### Pseudo Transitive Rule ($IR_6$)
In Pseudo transitive Rule, if X determines Y and YZ determines W, then XZ determines W.

If X → Y and YZ → W then XZ → W.

### DBMS Normalisation
A large database defined as a single relation may result in data duplication. This repetition of data may result in:
- Marketing relations very large.
- It isn't easy to maintain and update data as it would involve searching many records in relation.
- Wastage and poor utilization of disk space and resources.
- The likelihood of errors and inconsistencies increases.

So to handle these problems, we should analyze the decompose the relations with redundant data into smaller, simpler, and well-structured relations that are satisfy desirable properties. Normalisation is a process of decomposing the relations into relations with fewer attributes.

Data modification anomalies can be categorized into three types:
- **Insertion Anomaly**: Insertion Anomaly refers to when one cannot insert a new tuple into a relationship due to lack of data.
- **Deletion Anomaly**: The delete anomaly refers to the situation where the deletion of data results in the unintended loss of some other important data.
- **Updatation Anomaly**: The update anomaly is when an update of a single data value requires multiple rows of data to be updated.

#### Types of Normal Forms
Normalisation works through a serires of stages called Normal forms. The normal forms apply to individual relations. The relation is said to be in particular normal form if it satisfies constraints.

![](https://static.javatpoint.com/dbms/images/dbms-normalization.png)

| Normal Form | Description |
| ---- | ---- |
| 1NF | A relation is in 1NF if it contains an atomic value |
| 2NF | A relation will be in 2NF if it is in 1NF and all non-key attributes are fully functional dependent on the primary key |
| 3NF | A relation will be in 3NF if it is in 2NF and no transition dependency form |
| BCNF | A stronger defition of 3NF is known as Boyce Codd's normal form |
| 4NF | A relation will be in 4NF if it is in Boyce Codd's normal form and has no multi-valued dependency |
| 5NF | A relation is in 5NF. If it is in 4NF and does not contain any join dependency, joining should be lossless |

#### DBMS 1NF
- A relation will be 1NF if it contains an atomoci value.
- It states that an attribute of a table cannot hold multiple values. It must hold only single-valued attribute.
- First normal form disabllows the multi-valued attribute, composite attribute, and their combinations.

Example: Relation EMPLOYEE is not in 1NF because of multi-valued attribute EMP_PHONE.

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
- In the second normal form, all non-key attributes are fully functional dependent on the primary key.

Example: Let's assume, a school can store the data of teachers and the subjects they teach. In a school, a teacher can teach more than one subject.

| TEACHER_ID | SUBJECT | TEACHER_AGE|
| ---- | ---- | ---- | 
| 25 | Chemistry | 30| 
| 25 | Biology | 30| 
| 47 | English | 35| 
| 83 | Math | 38| 
| 83 | Computer | 38| 

In the given table, non-prime attribute TEACHER_AGE is dependent on TEACHER_ID which is a proper subset of a candidate key. That's why it violates the rule for 2NF.

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
- A relation will be in 3NF if it is in 2NF and not contain any transitive partial dependency.
- 3NF is used to reduce the data duplication. It is also used to achieve the data integrity.
- If there is no transitive dependency for non-prime attributes, then the relation must be in third normal form.

A relation is in third normal form if it holds atleast one of the following conditions for every non-trivial function dependency X → Y.
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

Here, EMP_STATE & EMP_CITY dependent on EMP_ZIP and EMP_ZIP dependent on EMP_ID. The non-prime attributes (EMP_STATE, EMP_CITY) transitively dependent on super key(EMP_ID). It violates the rule of third normal form.

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
- BCNF is the advance version of 3NF. It is stricter than 3NF.
- A table is in BCNF if every functional dependency X → Y, X is the super key of the table.
- For BCNF, the table should be in 3NF, and for every FD, LHS is super key.

Example: Let's assume there is a company where employees work in more than one department.

| EMP_ID | EMP_COUNTRY | EMP_DEPT | DEPT_TYPE | EMP_DEPT_NO |
| ---- | ---- | ---- | ---- | ---- |
| 264 | India | Designing | D394 | 283 |
| 264 | India | Testing | D394 | 300 |
| 364 | UK | Stores | D283 | 232 |
| 364 | UK | Developing | D283 | 549 |

In the above table Functional dependencies are as follows:
- EMP_ID  →  EMP_COUNTRY  
- EMP_DEPT  →   {DEPT_TYPE, EMP_DEPT_NO}  

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
- EMP_ID → EMP_COUNTRY  
- EMP_DEPT → {DEPT_TYPE, EMP_DEPT_NO}  

Candidate keys:
- For the first table: EMP_ID
- For the second table: EMP_DEPT
- For the third table: {EMP_ID, EMP_DEPT}

Now, this is in BCNF because left side part of both the functional dependencies is a key.

#### DBMS 4NF
- A relation will be in 4NF if it is in Boyce Codd normal form and has no multi-valued dependency.
- For a dependency A → B, if for a single value of A, multiple values of B exists, then the relation will be a multi-valued dependency.

| STU_ID | COURSE | HOBBY |
| ---- | ---- | ---- |
| 21 | Computer | Dancing | 
| 21 | Math | Singing | 
| 34 | Chemistry | Dancing | 
| 74 | Biology | Cricket | 
| 59 | Physics | Hockey | 

The given STUDENT table is in 3NF, but the COURSE and HOBBY are two independent entity. Hence, there is no relationship between COURSE and HOBBY.

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
- A relation is in 5NF if it is in 4NF and not contains any join dependency and joining should be lossless.
- 5NF is satisfied when all the tables are broken into as many tables as possible in order to avoid redundancy.
- 5NF is also known as Project-join normal form (PJ/NF).

| SUBJECT | LECTURER | SEMESTER | 
| ---- | ---- | ---- |
| Computer | Anshika | Semester 1 | 
| Computer | John | Semester 1 | 
| Math | John | Semester 1 | 
| Math | Akash | Semester 2 | 
| Chemistry | Praveen | Semester 1 | 

In the above table, John takes both Computer and Math class for Semester 1 but he doesn't take Math class for Semester 2. In this case, combination of all these fields required to identify a valid data.

Suppose we add a new Semester as Semester 3 but do not know about the subject and who will be taking that subject so we leave Lecturer and Subject as NULL. But all three columns together acts as a primary key, so we can't leave other two columns blank.

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

| SEMSTER | LECTURER | 
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
- When a relation in the relational model is not in appropriate normal form then the decomposition of a relation is required.
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
- For example, suppose there is a relation R (A, B, C, D) with functional dependency set (A->BC). The relational R is decomposed into R1(ABC) and R2(AD) which is dependency preserving because FD A->BC is a part of relation R1(ABC).

### Multivaled Depedency
- Multivalued dependency occurs when two attributes in a table are independent of each other but, both depend on a third attribute.
- A multivalued dependency consists of at least two attributes that are dependent on a third attribute that's why it always requires at least three attributes.

### Join Dependency
- Join decomposition is a further generalization of Multivalued dependencies.
- If the join of R1 and R2 over C is equal to relation R, then we can say that a join dependency (JD) exists.
- Where R1 and R2 are the decompositions R1(A, B, C) and R2(C, D) of a given relations R (A, B, C, D).
- Alternatively, R1 and R2 are a lossless decomposition of R.
- A JD ⋈ {R1, R2,..., Rn} is said to hold over a relation R if R1, R2,....., Rn is a lossless-join decomposition.
- The \*(A, B, C, D), (C, D) will be a JD of R if the join of join's attribute is equal to the relation R.
- Here, \*(R1, R2, R3) is used to indicate that relation R1, R2, R3 and so on are a JD of R.

### Inclusion Dependence
- Multivalued dependency and join dependency can be used to guide database design although they both are less common than functional dependencies.
- Inclusion dependencies are quite common. They typically show little influence on designing of the database.
- The inclusion dependency is a statement in which some columns of a relation are contained in other columns.
- The example of inclusion dependency is a foreign key. In one relation, the referring relation is contained in the primary key column(s) of the referenced relation.
- Suppose we have two relations R and S which was obtained by translating two entity sets such that every R entity is also an S entity.
- Inclusion dependency would be happen if projecting R on its key attributes yields a relation that is contained in the relation obtained by projecting S on its key attributes.
- In inclusion dependency, we should not split groups of attributes that participate in an inclusion dependency.
- In practice, most inclusion dependencies are key-based that is involved only keys.

### Canonical Cover
In the case of updating the database, the responsibility of the system is to check whether the existing functional dependencies are getting violated during the process of updating. In case of a violation of functional dependencies in the new database state, the rollback of the system must take place.

A canonical cover or irreducible a set of functional dependencies FD is a simplified set of FD that has a similar closure as the original set FD.

An attribute of an FD is said to be extraneous if we can remove it without changing the closure of the set of FD.

---

## Transaction Processing
### Transaction
### Transaction Property
### Stats of Transaction
### DBMS Schedule
### Testing of Serializability
### Conflict Schedule
### View Serializability
### Recoverability of Schedule
### Failure Classification
### Log-Based Recovery
### DBMS Checkpoint
### Deadlock in DBMS

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
