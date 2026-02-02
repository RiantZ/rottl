use rottl::get;

fn main() {
    let test = "world";
    let result = get(test);
    println!("{} {}", result, test);
}
