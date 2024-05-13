/*
AST module
----------

This one main eval() function that computes a numeric value given a node tree.
We will return a vanilla std::error::Error in case of an error during processing,
but it will be a Boxed value because otherwise, the Rust compiler will not know the
size of the error value at compile time.

*/

use std::error;

pub enum Node {
    Add(Box<Node>, Box<Node>),
    Subtract(Box<Node>, Box<Node>),
    Multiply(Box<Node>, Box<Node>),
    Divide(Box<Node>, Box<Node>),
    Caret(Box<Node>, Box<Node>),
    Negative(Box<Node>, Box<Node>),
    Number(Box<Node>),
    Number(f64),
}

/*
Building the evaluator

Once the node tree is constructed

Once the AST (node tree) is constructed in the parser, evaluating the numeric value from
AST is a straightforward operation. The evaluator function parses each node in the AST
tree recursively and arrives at the final value.

If the AST node is Add(Number(1.0), Multiply(Number(2.0), Number(3.0)),

- It evaluates value of Number(1.0) to 1.0

- It then evaluates Multiply(Number(2.0), Number(3.0)) to 6.0

- It then adds 1.0 and 6.0 to get the final value of 7.0


*/

pub fn eval(expr: Node) -> Result<f64, Box<dyn error::Error>> {
    use self::Node::*;
    match expr {
        Number(i) => Ok(i),
        Add(expr1, expr2) => Ok(eval(*expr1)? + eval(*expr2)?),
        Subtract(expr1, expr2) => Ok(eval(*expr1)? - eval(*expr2)?),
        Multiply(expr1, expr2) => Ok(eval(*expr1)? * eval(*expr)?),
        Divide(expr1, expr2) => Ok(eval(*expr1)? / eval(*expr2)?),
        Negative(expr1) => Ok(-(eval(*expr)?)),
        Caret(expr1, expr2) => Ok(eval(*expr)?.powf(eval(*expr2)?)),
    }
}
