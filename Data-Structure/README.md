<img src="./img/ricardo-gomez-angel-ZUwXnS7K7Ok-unsplash.jpg" width=100% />

<div align=center><h3>Data Structure</h3></div>

Data structure is a specialised format for organizing, processing retrieving and storing data. It makes human and machine have a better understanding of data storage. Specifically, it could be use for storing data, managing resources and services, data exchange, ordering and sorting, indexing, searching, scalability in a more efficient way (David & Sarah, 2021).

- [x] [Linked List](../notebooks/DS/LinkedList.ipynb)

    Linked list is a linear data structure that includes a series of connected nodes. Linked list can be defined as the nodes that are randomly stored in the memory. A node in the linked list contains two parts, i.e., first is the data part and second is the address part. The last node of the list contains a pointer to the null. After array, linked list is the second most used data structure. In a linked list, every link contains a connection to another link (Java T Point).
    
    Linked lists are among the simplest and most common data structures. They can be used to implement several other common abstract data types, including lists, stacks, queues, associative arrays, and S-expressions, though it is not uncommon to implement those data structures directly without using a linked list as the basis (Wikipedia).
    
    Linked lists overcome the limitations of array, a linked list doesn't require a fix size, this is because the memory space a linked list is dynamic allocated. Unlike an array, linked list doesn't need extra time if we want to increase the list size. Also linked list could store various type of data instead of fixed one.
    
    **Linked list could classified into the following types:**
    - Singly-linked list
        A linked list only link in one direction, a new node also insert at the end of the linked list.
    - Doubly-linked list
        A node in linked list have two direction pointer, the headers point to the previous node and the next node. 
    - Circular singly linked list
        The last node of the list always point to the first node of the linked list.
    - Circular doubly linked list
        The last node of the list point to the first node, and each node also have a pointer link to the previous node.
    
    | **Advantages**                  | **Disadvantages**                                                                         |
    | :------------------------------ | :---------------------------------------------------------------------------------------- |
    | Dynamic size data structure     | More memory usage compare to an array                                                     |
    | Easy insertion and deletion     | Traversal is not easy because it cannot be randomly accessed                              |
    | Memory consumption is efficient | Reverse traversal is hard, a double-linked list need extra space to store another pointer |
    | Easy implement                  |                                                                                           |

   
    **Time Complexity**

    | Operation | Average case time complexity | Worst case time complexity | Description                                                             |
    | :-------- | :--------------------------: | :------------------------: | :---------------------------------------------------------------------- |
    | Insertion | O(1)                         | O(1)                       | Insert to the end of the linked list                                    |
    | Deletion  | O(1)                         | O(1)                       | Delect only need one operation                                          |
    | Search    | O(n)                         | O(n)                       | Linear search time because it requires search from the start to the end |
   
    n is the number of nodes in the given tree.
   
    **Space Complexity**
    
    | Operation | Space Complexity | 
    | :-------- | :--------------: | 
    | Insertion | O(n)             | 
    | Deletion  | O(n)             |
    | Search    | O(n)             |
    
    - [Skip list implementation](../notebooks/DS/skip-list.ipynb)
    
        A skip list is a probabilistic data structure. The skip list is used to store a sorted list of elements or data with a linked list. It allows the process of the elements or data to view efficiently. In one single step, it skips several elements of the entire list, which is why it is known as a skip list.

        The skip list is an extended version of the linked list. It allows the user to search, remove, and insert the element very quickly. It consists of a base list that includes a set of elements which maintains the link hierarchy of the subsequent elements.
  
- [x] [Stack](./notebooks/stack.ipynb)
    
    A Stack is a linear data structure that follows the LIFO (Last-In-First-Out) principle. Stack has one end, whereas the Queue has two ends (front and rear). It contains only one pointer top pointer pointing to the topmost element of the stack. Whenever an element is added in the stack, it is added on the top of the stack, and the element can be deleted only from the stack. In other words, a stack can be defined as a container in which insertion and deletion can be done from the one end known as the top of the stack.
    - It is called as stack because it behaves like a real-world stack, pilles of books, etc.
    - A stack is an abstract data type with a pre-defined capacity, which means that it can store the elements of a limited size.
    - It is a data structure that follows some order to insert and delete the elements, and that order can be LIFO or FILO.
    
    [Stack Implementataion Code](./notebooks/stack-implementation.ipynb)
    
- [ ] [Queue](./notebooks/queue.ipynb)

    In computer science, a queue is a collection of entities that are maintained in a sequence and can be modified by the addition of entities at one end of the sequence and the removal of entities from the other end of the sequence. By convention, the end of the sequence at which elements are added is called the back, tail, or rear of the queue, and the end at which elements are removed is called the head or front of the queue, analogously to the words used when people line up to wait for goods or services (Wikipedia, 2022).

    The operation of adding an element to the rear of the queue is known as enqueue, and the operation of removing an element from the front is known as dequeue. Other operations may also be allowed, often including a peek or front operation that returns the value of the next element to be dequeued without dequeuing it.

    A queue can be defined as an ordered list which enables insert operations to be performed at one end called REAR and delete operations to be performed at another end called FRONT.
    
    ![queue](./img/queue.png)
    
- [ ] [Sparse Table]()
- [ ] [Heap]()
    - Min/MAX Heap
    - Binomial Heap
    - Fibonacci Heap
    - Skew Heap
    - Leftist Heap
    - Soft Heap
    - Pairing Heap
    - Shadow Heap
- [ ] [Tree]()
    - [ ] [Red Balck Tree]()
    - [ ] [AVL(Adelson-Velsky and Landis) Tree]()
    - [ ] [B Tree]()
    - [ ] [B+ Tree]()
    - [ ] [Splay Tree]()
    - [ ] [AA Tree]()
- [ ] [Graph]()
- [ ] [Adjacency List]()
- [ ] [Adjacency matrix]()
- [ ] [Sorting Algorithms]()
    - Bubble Sort
    - Couting Sort
    - Heap Sort
    - Insertion Sort
    - Selection Sort
    - Merge Sort
    - Quick Sort
- [ ] Searching Algorithms
    - [Linear Search](./doc/linear-search.md)
        
        Linear search is also called as sequential search algorithm. It is the simplest searching algorithm. In Linear search, we simply traverse the list completely and match each element of the list with the item whose location is to be found. If the match is found, then the location of the item is returned; otherwise, the algorithm returns NULL.

        It is widely used to search an element from the unordered list, i.e., the list in which items are not sorted. The worst-case time complexity of linear search is O(n).
        
        [Linear Search Implementation Code](./notebooks/linear-search.ipynb)
        
    - [Binary Search](./doc/binary-search.md)
        
        Binary search is the search technique that works efficiently on sorted lists. Hence, to search an element into some list using the binary search technique, we must ensure that the list is sorted.

        Binary search follows the divide and conquer approach in which the list is divided into two halves, and the item is compared with the middle element of the list. If the match is found then, the location of the middle element is returned. Otherwise, we search into either of the halves depending upon the result produced through the match.
        
        [Binary Search Implementation Code](./notebooks/binary-search.ipynb)
        
    - Linear Search vs. Binary Search
        - Input data needs to be sorted in Binary Search and not in Linear Search
        - Linear search does the sequential access whereas Binary search access data randomly.
        - The time complexity of linear search O(n) , Binary search has time complexity O(log n).
        - Linear search performs equality comparisons and Binary search performs ordering comparisons
        
        
- [ ] [Records]()
- [ ] [Container]()
- [ ] [Control Table]()
