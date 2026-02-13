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
