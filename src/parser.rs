use super::token::OperPrec; // import the operator precedence enu
use super::token::Token; // import the token enum
use super::tokenizer::Tokenizer;

/*

Dealing with Errors
-------------------

In our project, errors can occur due to two main reasons - there could be a programming
error, or an error could occur due to invalid inputs. Let's first discuss the Rust approach to
error handling.

In rust, errors are first-class citizens in that an error is a data type in itself, just like
an integer, string or vector. Because error is a data type, type checking can happen
at compile time. The rust standard library has a std::error::Error trait implemented
by all errors in the Rust standard library.

Result<T, E> is an enum with two variants, where Ok(T) represents success and
Err(E) represents the error returned. Pattern matching is used to handle
the two types of return values from a function


*/

pub enum ParserError {
    UnableToParse(String),
    InvalidOperator(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            self::ParseError::UnableToParse(e) => write!(f, "Error in evaluating {}", e),
            self::ParseError::InvalidOperator(e) => write!(f, "Error in evaluating {}", e),
        }
    }
}

pub struct Parser<'b> {
    tokenizer: Tokenizer<'b>, // the tokenizer cannot outlive the peekable reference
    current_token: Token,
}

// implement methods for this parser
impl<'b> Parser<'b> {
    /*

    Parser methods
    --------------
    The parser struct will have two public methods

    new(): To create a new instance of the parser. This new() method will create
    a tokenizer instance passing in the arithmetic expression, and then stores
    the first token

     */

    pub fn new(expr: &'a str) -> Result<Self, ParserError> {
        let mut lexer = Tokenizer::new(expr); // creates a struct with implementations
                                              /*
                                              Creates an instance of Tokenizer, intializing it with the arithmetic
                                              expression, and then tries to retrieve the first token from the expression
                                               */
        let cur_token = match lexer.next() {
            Some(token) => token,
            None => return Err(ParseError::InvalidOperator("Invalid character".into())),
        };

        // When a non-error, return the parser struct initialized
        Ok(Parser {
            tokenizer: lexer,
            current_token: cur_token,
        })
    }

    pub fn parse(&mut self) -> Result<Node, ParseError> {
        /*
        The following code is for the parse() method. It invokes
            a private generate_ast() method that does the processing recursively
            and returns a AST. If successful, it returns the node tree. It not,
            it propagates the errors received
         */
        let ast = self.generate_ast(OperPrec::DefaultZero);
        match ast {
            Ok(ast) => Ok(ast),
            Err(e) => Err(e),
        }
    }

    fn get_next_token(&mut self) -> Result<(), ParseError> {
        /*
            This is a private method that is used
            by the other public methods

        Result<(), ParseError> - if nothing goes wrong,
        no concrete value is returned
            */
        let next_token = match self.tokenizer.next() {
            Some(token) => token,
            None => return Err(ParseError::InvalidOperator("Invalid character".into())),
        };
        // if we pass the above error functionalities, return the
        // current token as the next token from 'now'
        self.current_token = next_token;
        Ok(())
    }

    fn check_paren(&mut self, expected: Token) -> Result<(), ParseError> {
        if expected == self.current.token {
            self.get_next_token()?;
            Ok(())
        } else {
            Err(ParseError::InvalidOperator(format!(
                "Expected {:?}, got {:?}",
                expected, self.current_token
            )))
        }
    }

    fn parse_number(&mut self) -> Result<Node, ParseError> {
        /*
            The parse_number() method takes the current token, and checks for three things

            - Whether the token is a number of the form Num(i)

        - Whether the token as a sign, in case it is a negative number. For example,
          the expression -2.2 + 3.4 is parsed into AST as Add(Negative(Number(2.))
             */

        let token = self.current_token.clone();

        match token {
            Token::Subtract => {
                self.get_next_token()?;
                let expr = self.generate_ast(OperPrec::Negative)?;
                Ok(Node::Negative(Box::new(expr)))
            }

            Token::Num(i) => {
                self.get_next_token()?;
                Ok(Node::Number(i))
            }

            Token::LeftParam => {
                self.get_next_token()?;
                let expr = self.generate_ast(OperPrec::DefaultZero)?;
                self.check_paren(Token::RightParen)?;

                if current_token == Token::LeftParen {
                    let right = self.generate_ast(OperPrec::MulDiv)?;

                    return Ok(Node::Multiply(Box::new(expr), Box::new(right)));
                }
                Ok(expr)
            }

            _ => Err(ParseError::UnableToParse("Unable to parse".to_string())),
        }
    }

    fn generate_ast(&mut self, oper_prec: OperPrec) -> Result<Node, ParseError> {
        /*
            The generate_ast() method is the main workhorse of the module and is invoked
            recursively. It does it's processing in the following sequence.

            1. It processes numeric tokens, negative number tokens, and expressions
            in parentheses using the parse_number() method

            2. It parses each token from the arithmetic expression in a sequence within
           a loop to check

        */

        let mut left_expr = self.parse_number()?;

        while oper_prec < self.current_token.get_oper_prec() {
            if self.current_token == Token::EOF {
                break;
            }

            let right_expr = self.convert_token_to_node(left_expr.clone())?;
            left_expr = right_expr;
        }

        Ok(left_expr)
    }

    fn convert_token_to_node(&mut self, left_expr: Node) -> Result<Node, ParseError> {
        match self.current_token {
            Token::Add => {
                self.get_next_token()?;
                // Get the right-side expression

                let right_expr = self.generate_ast(OperPrec::AddSub)?;
                Ok(Node::Add(Box::new(left_expr), Box::new(right_expr)))
            }

            Token::Subtract => {
                self.get_next_token()?;
                // Get right-side expression
                let right_expr = self.generate_ast(OperPrec::AddSub)?;
                Ok(Node::Subtract(Box::new(left_expr), Box::new(right_expr)))
            }

            Token::Multiply => {
                self.get_next_token()?;
                // Get right side expression
                let right_expr = self.generate_ast(OperPrec::MulDiv)?;

                Ok(Node::Multiply(Box::new(left_expr), Box::new(right_expr)))
            }

            Token::Divide => {
                // if we can get a  next token,
                // then generate the the ast
                self.get_next_token()?;
                // Get right side expression
                let right_expr = self.generate_ast(OperPrec::MulDiv)?;
                Ok(Node::Divide(Box::new(left_expr), Box::new(right_expr)))
            }

            Token::Caret => {
                self.get_next_token()?;
                // Get right side expression
                let right_expr = self.generate_ast(OperPrec::Power)?;
                Ok(Node::Caret(Box::new(left_expr)))
            }

            _ => Err(ParseError::InvalidOperator(format!(
                "Please enter valid operator {:?}",
                self.current_token
            ))),
        }
    }
}
