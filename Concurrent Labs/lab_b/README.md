# 600086-Lab-B

This lab introduces basic elements of threading. As you progress through this semester, you'll be learning a lot about parallel and concurrent programming, that will hopefully prove valuable both during your career and also as technical background for the likely questions you'll face in interviews.

To assist you in remembering these intricacies you'll be using a Lab Book to record your observations and findings, solutions to common problems, and even how you solved a problematic bug.

## GIT

All the source code for this lab exercise is on GitHub

## Lab Book

An example of the Lab book can be found on the GIT within this Lab

## Q1. First threads

Open the folder **first_thread**, in VS Code and run `cargo init` in the **terminal window**.

Open the main Rust file and edit the auto-generated Rust code.  Move the print "hello world" to a new function called `my_function`, and then call `my_function` from the main program.

Look at the lecture material for guidance on function syntax.

Now replace the synchronous call to your function with an asynchronous call i.e. create a thread with your function set as its thread Function.
Use the following snippets of Rust as a guide:

```Rust
    std::thread::spawn(move || my_function() );
```

Now expand your code to include a second thread, which calls a second function.  Add a print statement to this second function.
Notice how occasionally the output is jumbled.  This is a race condition, caused by a thread unsafe print method.  We'll address these topics in a future lecture.

Add a sleep method to one of your functions.  Sleep is available in the thread library.

Use the following snippet of Rust as a guide:

```Rust
    std::thread::sleep(std::time::Duration::new(5, 0));
```

## Q2. Joining threads

Open the folder **joining_threads**, in VS Code and run `cargo init` in the **terminal window**.

Copy the code from the previous exercise into `main.rs`

Alter the code to create N threads rather than two threads, where N is a arbitrary number.  Use a for loop to achieve this.

Use the following snippets of Rust as a guide, where id is the loop counter and `num_of_threads` the loop limit:

```Rust
    for id in 0..num_of_threads {
        // Add code here
    }
```

Now that you can create an arbitrary number of threads, the next step is to ensure that all threads terminate before the application (the primary thread) terminates.  To do this we can use a join method.
The join method blocks until the thread has terminated.

```Rust
t.join();
```

However, in order to use the join method you need to hold a reference to each thread.  To do this, use a list to store the threads when they are created.  Then iterate over the list to ensure they are all joined.
The following snippets of Rust show the basics of list manipulation.

```Rust
    let mut list_of_threads = vec!();

    list_of_threads.push(t);

    for t in list_of_threads {
        // Add code here
    }
```

Remember Rust variables are immutable (constant) by default.  Use the `mut` keyword to make them mutable.

Add a print statement at the end of the main section of code and check that all threads have terminated prior to seeing this print message.

## Q3. Experimentation

Now that you have the basic framework for creating and joining threads, experiment with giving the threads items of work, as well as altering the number of threads used.

Use **Windows Task Manager** to observe your program running on the CPU.

Info on each core is available on the **Performance** tab, then right click and change graph to **logical processors**
