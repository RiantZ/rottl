use rottl::Lexer;

fn main() {
    let input = r#"set(attributes["service.name"], "myapp") where status == 200"#;

    println!("Input: {}", input);
    println!("\nTokens:");

    for token in Lexer::collect_tokens(input) {
        println!("  {:?}", token);
    }
}
