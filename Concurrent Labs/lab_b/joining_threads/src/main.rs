fn my_function(id: usize) {
    while true {
        print!("id: {id}\n");
    }
}

fn main() {
    let mut threads = vec![];
    for id in 0..1_000_000_000 {
        threads.push(std::thread::spawn(move || my_function(id)))
    }

    for thread in threads {
        thread.join().expect("Thread panic");
    }
}
