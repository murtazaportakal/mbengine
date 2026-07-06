fn main() {
    let editor = engine::app::node_editor::NodeGraphEditor::new();
    let code = editor.compile_to_rhai();
    println!("--- COMPILED RHAI CODE ---");
    println!("{}", code);
    println!("--------------------------");
}
