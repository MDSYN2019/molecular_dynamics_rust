#[derive(Debug, PartialEq, PartialOrd)]
// Defines all the OperPrec levels, from the lowest to highest
pub enum OperPrec {
    DefaultZero,
    AddSub,
    MulDiv,
    Power,
    Negative,
}

#[derive(Debug, PartialEq, PartialOrd)]
pub enum Token {
    Add, // + Add token is generated when + is encountered
    Subtract,
    Multiply,
    Divide,
    Caret,
    LeftParen,
    RightParen,
    Num(f64),
    EOF,
}

impl Token {
    /*
    implement methods for the token enum
    */
    pub fn get_oper_prec(&self) -> OperPrec {
        use self::OperPrec::*;
        use self::Token::*;
        match *self {
            Add | Subtract => AddSub,
            Multiply | Divide => MulDiv,
            Caret => Power,
            _ => DefaultZero,
        }
    }
}
