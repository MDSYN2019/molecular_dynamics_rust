use super::token::token; // import token module here
use std::iter::Peekable;

pub struct Tokenizer<'a> {
    expr: Peekable<Chars<'a>>, // the tokenizer cannot outlive the peekable reference
}

/*
Tokenizer module:

This has two public methods:

new() and next(). The new method is fairly simple and just creates
a new instance of the Tokenzier struct and initiates it.

No error will be returned in this method. However, the next() method
returns a token, and if there is any invalid character in the arithmetic
character in the arithmetic expression, we need to deal with the situation
and communicate it to the calling code.


*/
impl<'a> Tokenizer<'a> {
    pub fn new(new_expr: &'a str) -> Self {
        Tokenizer {
            expr: new_expr.chars().peekable(),
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token; // Can be Add, subtract etc..
    fn next(&mut self) -> Option<Token> {
        let next_char = self.expr.next(); // returns the next character
        match next_char {
            /*
            The returned character is then evaluated using a match statement. Pattern matching
            is used to determine what token to return, depending on what character.
             */
            Some('0'..='9') => {
                let mut number = next_char?.to_string();
                while let Some(next_char) = self.expr.peek() {
                    if next_char.is_numeric() || next_char == &'.' {
                        number.push(self.expr.next()?);
                    } else if next_char == &'(' {
                        return None;
                    } else {
                        break;
                    }
                }
                // we are unwrapping, confident that we have passed
                // the error conditional
                Some(Token::Num(number.parse::<f64>().unwrap()))
            }
            Some('+') => Some(Token::Add),
            Some('-') => Some(Token::Subtract),
            Some('*') => Some(Token::Multiply),
            Some('/') => Some(Token::Divide),
            Some('^') => Some(Token::Caret),
            Some('(') => Some(Token::LeftParen),
            Some(')') => Some(Token::RightParen),
            None => Some(Token::EOF),
            Some(_) => None,
        }
    }
}
