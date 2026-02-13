# LAB RUST WEEK TWO
# Question 1. First Thread
Create two functions that both print a message and spawn threads for each one
## Solution
```rs
fn my_function() {
    println!("Hello, world!");
}

fn my_other_function() {
    println!("Greetings, world!");
}

fn main() {
    std::thread::spawn(move || my_function());
    std::thread::spawn(move || my_other_function());
    std::thread::sleep(std::time::Duration::new(5, 0));
}
```
## Output
```
Hello, world!
Greetings, world!
```
```
Greetings, world!
Hello, world!
```
Either output shows up at random depending on the mood of the scheduler.
## Reflection
For this lab it shows off how to spawn threads in Rust. This means each of the functions will be running concurrently in their own thread. 

The operating system will schedule them to run on their own CPU thread or share time on the CPU. This means the order of execution is not guaranteed and can vary each time the program is run as the scheduler will not consistently schedule work in the same way. 

We are told that we should expect the output to be jumbled however I could not reproduce it. Just that the order of the output is not guaranteed.

# Question 2. Joining Threads
Use the `.join()` method to wait for an arbituary amount of threads to finish instead of telling the main thread to sleep.

## Solution
```rs
fn my_function(id: usize) {
    println!("Hello from thread {id}");
}

fn main() {
    let mut threads = vec![];
    for id in 0..10 {
        threads.push(std::thread::spawn(move || my_function(id)))
    }

    for thread in threads {
        thread.join().expect("Thread panic");
    }
}
```

## Output
```
Hello from thread 1
Hello from thread 2
Hello from thread 5
Hello from thread 4
Hello from thread 3
Hello from thread 0
Hello from thread 6
Hello from thread 7
Hello from thread 8
Hello from thread 9
```
## Reflection
When spawning a thread it returns a JoinHandle of which `.join()` can be called on. `.join()` will wait for the thread to finish before returning, and will return instantly if already finished.

We can use a vector to store multiple JoinHandles and then loop over them to wait for all threads to finish without having to set an arbitrary sleep duration.

# Q3. Experimentation
Play around with the number of threads and tasks of the threads and monitor the CPU usage.

## Stuff i did
### Thing one

I created a while true loop that prints the id over and over and created 16 threads (one for each system thread)


#### What happened


It didn't max out each cpu core as i thought would happen. It use like 20% of each core with one or two core boosting up to like 60%-70%.
The output shows each thread getting about 5-20 cycles of printing then switching to another thread


#### Reflection


This is likely the systems scheduler doing its job to ensure that each thread gets a share of CPU time, while not starving other processes.


### Thing two
I created a while true loop that prints the id over and over and created 1_000_000_000 threads
#### What happened

The CPU usage dramatically increased with most cores increasing to 50ish% and one or two being maxed out. The system would then freeze momentarily before the rust process prematurely exits with no warning.
#### Reflection

I first thought that the system would be running out of memory but monitoring that it never drops below 20gb available, so theres plent of memory.

My next assumption is as more threads get spawned the system eventually says it can't make any more threads and then it kills the rust process.
Looking into thread limits, on my system with `prlimit -u` there seems to be a hard cap of processes at 127089 which from looking at my output I could not see a thread higher than 100_000. So it likely dies when it hits that limit.
