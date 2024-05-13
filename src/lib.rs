/*
Code based on the following paper:

https://apt.scitation.org/doi/abs/10.1119/10.0002644?journalCode=ajp

Title: A first encounter with the Hartree-fock self-consistent field method

*/

extern crate assert_type_eq;

pub mod tokenizer {
    use std::iter::Peekable;
    use std::num::ParseIntError;
    use std::str::Chars; // import the token enum

    pub struct A; // A concrete type A
    pub struct SingleGen<T>(T);

    pub struct Sheep {
        naked: bool,
        name: &'static str,
    }

    pub trait Animal {
        // Associated function signature: self
        fn new(name: &'static str) -> Self;
        // Method signature: these will return a string
        fn name(&self) -> &'static str;
        fn noise(&self) -> &'static str;
        // traits can provide default method definitions
        fn talk(&self) {
            println!("{} says {}", self.name(), self.noise())
        }
    }

    impl Sheep {
        fn is_naked(&self) -> bool {
            self.naked
        }

        fn shear(&mut self) {
            if self.is_naked() {
                println!("{} is already naked..", self.name());
            } else {
                println!("{} gets a haircut!", self.name);
                self.naked = true;
            }
        }
    }

    impl Animal for Sheep {
        // implement the Animal trait for Sheep
        fn new(name: &'static str) -> Sheep {
            Sheep {
                name: name,
                naked: false,
            }
        }
        fn name(&self) -> &'static str {
            self.name
        }
        fn noise(&self) -> &'static str {
            if self.is_naked() {
                "baaaah?"
            } else {
                "baaaah!"
            }
        }
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

    pub fn multiply(first_number_str: &str, second_number_str: &str) -> Result<i32, ParseIntError> {
        match first_number_str.parse::<i32>() {
            Ok(first_number) => match second_number_str.parse::<i32>() {
                Ok(second_number) => Ok(first_number * second_number),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        }
    }

    pub fn divide(numerator: f64, denominator: f64) -> Option<f64> {
        if denominator == 0.0 {
            None
        } else {
            Some(numerator / denominator)
        }
    }

    // operator precedence enum

    #[derive(Debug, PartialEq, PartialOrd)]
    // defines all the operprec levels, from the lowest to the highest
    pub enum OperPrec {
        DefaultZero,
        AddSub,
        MulDiv,
        Power,
        Negative,
    }

    pub struct Tokenizer<'a> {
        expr: Peekable<Chars<'a>>, // the tokenizer cannot outlive the peekable reference
    }

    impl<'a> Tokenizer<'a> {
        pub fn new(new_expr: &'a str) -> Self {
            Tokenizer {
                expr: new_expr.chars().peekable(),
            }
        }
    }

    impl<'a> Iterator for Tokenizer<'a> {
        type Item = Token;

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

    /*

    Building the parser
    -------------------

    The parser is the module in our project that constructs the AST, which is a tree of nodes
    with each node representing a token (a number or an arithmetic oeprator). The AST is
    a recursive tree structure of token nodes, the root node is a token, which contains
    child nodes that are also tokens

    The parser uses the tokenizer outputs to construct an overall AST, which is a heirachy of nodes.
    The structure of AST constructed from the parser is illustrated.

    ---
    Each of these nodes is stored in a boxed data structure, which means the actual data value
    for each node is stored in heap memory, while the pointer to each of the nodes is stored a box variable
    ---
     */

    pub enum Node {
        Add(Box<Node>, Box<Node>),
        Subtract(Box<Node>, Box<Node>),
        Multiply(Box<Node>, Box<Node>),
        Divide(Box<Node>, Box<Node>),
        Caret(Box<Node>, Box<Node>),
        Negative(Box<Node>),
        Number(f64),
    }

    pub enum ParseError {
        UnableToParse(String),
        InvalidOperator(String),
    }

    pub struct Parser<'b> {
        expr: Tokenizer<'b>, // the tokenizer cannot outlive the peekable reference
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

        pub fn new(expr: &'b str) -> Result<Self, ParseError> {
            let mut lexer = Tokenizer::new(expr);
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
                expr: lexer,
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
            let next_token = match self.expr.next() {
                Some(token) => token,
                None => return Err(ParseError::InvalidOperator("Invalid character".into())),
            };
            // if we pass the above error functionalities, return the
            // current token as the next token from 'now'
            self.current_token = next_token;
            Ok(())
        }

        fn check_paren(&mut self, expected: Token) -> Result<(), ParseError> {
            if expected == self.current_token {
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
            let token = &self.current_token;
            /*
               The parse number() method takes the current token,
               and checks for three things

               - Whether the token is a number of the form Num(i)

               - Whether the token has a sign
            */
            match token {
                Token::Subtract => {
                    self.get_next_token()?;
                    let expr = self.generate_ast(OperPrec::Negative)?;
                    Ok(Node::Negative(Box::new(expr)))
                }

                //Token::Num(i) => {
                //    self.get_next_token()?;
                //    Ok(Node::Number(i))
                //}
                Token::LeftParen => {
                    self.get_next_token()?;
                    let expr = self.generate_ast(OperPrec::DefaultZero)?;
                    self.check_paren(Token::RightParen)?;

                    if self.current_token == Token::LeftParen {
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
                let right_expr = self.convert_token_to_node(left_expr)?;
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
                    Ok(Node::Caret(Box::new(left_expr), Box::new(right_expr)))
                }

                _ => Err(ParseError::InvalidOperator(format!(
                    "Please enter valid operator {:?}",
                    self.current_token
                ))),
            }
        }
    }
}

//pub fn bond_angles(readable_file: &str, i: usize, j: usize) -> DataFrame {
//    /*
//    if we have a index i and j indicating the row indices, make a function that
//    lo1oks takes those indices and prints out the angles between them
//     */
//    let df = polars_read_molecular_data_file(readable_file);
//    df
//}

//pub fn polars_compute_bond_rij(x_column: &str, y_column: &str, z_column: &str) -> () {
//    /*
//    We are trying to compute the interatomic distance between all atoms Rij
//     */
//    let mut molecular_data = polars_read_molecular_data_file(readable_file)
//        .expect("Failed to read the molecular data");
//
//    let df_numerical = df
//        .clone()
//        .lazy()
//        .select([
//            (col("nrs") + lit(5)).alias("nrs + 5"),
//            (col("nrs") - lit(5)).alias("nrs - 5"),
//            (col("nrs") * col("random")).alias("nrs * random"),
//            (col("nrs") / col("random")).alias("nrs / random"),
//        ])
//        .collect()?;
//}
//
//pub fn polars_compute_bond_angles(x_column: &str, y_column: &str, z_column: &str) -> () {
//    /*
//    Compute the bond angles with the x, y and z data
//     */
//    let x_column_data = output.clone().lazy().select([col(x_column)]).collect()?;
//    let y_column_data = output.clone().lazy().select([col(y_column)]).collect()?;
//    let z_column_data = output.clone().lazy().select([col(z_column)]).collect()?;
//}

// Main executable function
//fn open_xyz(geom_file: str) {
//    /*
//    opening the xyz file as necessary
//    */
//    let StructureFilename = String::from(geom);
//    let coordinateInformation = fs::read_to_string(St1ructureFilename);
//    let xy = match coordinateInformation {
//        // this needs to be figured out now
//        Ok(content) => content,
//        Err(error) => {
//            panic!("Could not open or find file: {}", error);
//        }
//    };
//}

pub mod tensors {
    pub fn outer_product<T>(a: &[T], b: &[T], default_value: T) -> Vec<Vec<T>>
    where
        T: std::ops::Mul<Output = T> + Clone,
        /*
        The function takes two slices &[T] as input. Slices are references
        to arrays or vectors, allowing you to work with portion of a collection,
        to return a vector of vectors of type T
         */
    {
        let mut result = vec![vec![default_value.clone(); b.len()]; a.len()];
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i][j] = a[i].clone() * b[j].clone();
            }
        }
        result
    }
}

pub mod periodic_boundary_conditions {
    /*
    How do we handle periodic boundaries and minimum image convention in a simulation program?

    Define simulation box and write the necessary methods
    to fix the coordiantes when the molecule has periodic boundary issues

     */
    pub struct SimulationBox {
        pub x_dimension: f64,
        pub y_dimension: f64,
        pub z_dimension: f64,
    }

    pub struct MolecularCoordinates {}
}

pub mod lennard_jones_simulations {
    use ndarray::Array2;
    use rand::prelude::*;
    use rand::Rng;
    use std::error::Error;

    pub struct LJParameters {
        pub nsteps: i64,
        pub n: i64,
        pub i: i64,
        pub j: i64,
        pub eps: f64,
        pub sigma: f64,
        pub sigma_sq: f64,
        pub pot: f64,
        pub rij_sq: f64,
        pub sr2: f64,
        pub sr6: f64,
        pub sr12: f64,
        pub epslj: f64,
        pub na: i64,
    }

    pub struct Particle {
        position: (f64, f64, f64),
        velocity: (f64, f64, f64),
        lj_parameters: LJParameters,
    }

    impl Particle {
        fn maxwellboltzmannvelocity(&mut self, temp: f64, mass: f64, v_max: f64) -> () {
            /*
            A temperature bathc can be achieved by periodically resetting
            all velocities from the Maxwell-Boltzmann distribution
            at the desired temperature

            More information here:

            https://scicomp.stackexchange.com/questions/19969/how-do-i-generate-maxwell-boltzmann-variates-using-a-uniform-distribution-random

            -----

            Basically, a function to initalize the velocities of the particles within
            and to try to create a initial velocity pool that is consisten with the given temperature


             */
            let mut rng = rand::thread_rng();
            let sigma_mb = (temp / mass).sqrt();
            let mut velocities: Vec<usize> = Vec::with_capacity(self.lj_parameters.na as usize);
            // might be worth making a function here to make it testable
            let v_x = rng.gen::<f64>() * 2.0 * v_max - v_max;
            let v_y = rng.gen::<f64>() * 2.0 * v_max - v_max;
            let v_z = rng.gen::<f64>() * 2.0 * v_max - v_max;
            println!("velocities are {:?} {:?} {:?}", v_x, v_y, v_z);

            let prob = (-0.5 * (v_x * v_x + v_y * v_y + v_z * v_z)
                / (self.lj_parameters.sigma * self.lj_parameters.sigma))
                .exp();
            let rand_val: f64 = rng.gen();

            if rand_val < prob {
                self.velocity.0 = rng.gen::<f64>() * 2.0 * v_max - v_max;
                self.velocity.1 = rng.gen::<f64>() * 2.0 * v_max - v_max;
                self.velocity.2 = rng.gen::<f64>() * 2.0 * v_max - v_max;
            }
        }

        fn update_position_verlet(&mut self, acceleration: (f64, f64, f64), dt: f64) -> () {
            /*
            verlet scheme to change the position
            */
            self.position.0 += self.velocity.0 * dt + 0.5 * acceleration.0 * dt * dt;
            self.position.1 += self.velocity.1 * dt + 0.5 * acceleration.1 * dt * dt;
            self.position.2 += self.velocity.2 * dt + 0.5 * acceleration.2 * dt * dt;
        }

        fn update_velocity_verlet(
            &mut self,
            old_acceleration: (f64, f64, f64),
            new_acceleration: (f64, f64, f64),
            dt: f64,
        ) {
            self.velocity.0 += 0.5 * (old_acceleration.0 + new_acceleration.0) * dt;
            self.velocity.1 += 0.5 * (old_acceleration.1 + new_acceleration.1) * dt;
            self.velocity.2 += 0.5 * (old_acceleration.2 + new_acceleration.2) * dt;
        }
    }

    pub fn create_atoms_with_set_positions_and_velocities(
        number_of_atoms: i64,
        temp: f64,
        mass: f64,
        v_max: f64,
    ) -> Result<Vec<Particle>, Box<dyn Error>> {
        /*
        	*/
        let mut rng = thread_rng();
        let mut vector_positions: Vec<Particle> = Vec::new();
        for _ in 0..number_of_atoms {
            let mut particle = Particle {
                position: (
                    // generate x y z position values between -10 and 10
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(-10.0..10.0),
                ),
                velocity: (
                    // generate velocity values between -1 and 1
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                ),

                lj_parameters: (LJParameters {
                    n: 10,
                    i: 0,
                    j: 1,
                    eps: 1.0,
                    sigma: 1.0,
                    sigma_sq: 0.0,
                    pot: 0.0,
                    rij_sq: 0.0,
                    sr2: 0.0,
                    sr6: 0.0,
                    sr12: 0.0,
                    epslj: 0.0,
                    nsteps: 1000,
                    na: 1,
                }),
            };
            // Reset the positions to the maxwell boltzmann distibution of velocities
            particle.maxwellboltzmannvelocity(temp, mass, v_max);
            // push those values into the vector
            vector_positions.push(particle);
        }
        Ok(vector_positions)
    }

    pub fn run_verlet_update(
        mut particles: Vec<Particle>,
        acceleration: (f64, f64, f64),
        dt: f64,
    ) -> () {
        /*
        Update the position and velocity of the particle using the verlet scheme
        */
        // update the position
        for particle in particles.iter_mut() {
            particle.update_position_verlet(acceleration, dt);
            // update the velocity
            particle.update_velocity_verlet(acceleration, acceleration, dt);
        }
    }

    pub fn run_md_nve() {
        // create the particles as necessary - matching
        // the nve simulatons

        /*
        We are now equipt to implement a NVE molecular dynamics simulations.
        */
        let mut particles = create_atoms_with_set_positions_and_velocities(10, 10.0, 10.0, 10.0);
        // First update the positions
    }

    impl LJParameters {
        /*
        Define the methods for the lennard-jones method
        */

        pub fn lennard_jones(&mut self, sigma: f64, r: f64, epsilon: f64) -> f64 {
            /*
            Private method

            Return the standard lennard jones function
             */

            let u_ij = 4. * epsilon * (f64::powi(sigma / r, 12) - f64::powi(sigma / r, 6));
            u_ij
        }

        pub fn hard_sphere(&mut self, sigma: f64, r: f64, epsilon: f64) -> f64 {
            /*
            Private method

            Return the hard-sphere potential
             */
            let mut u_ij = 0.0;

            if r < sigma {
                u_ij = 1000000000000000000000.0; // meant to simulate infinity..
            } else {
                u_ij = 0.0; // need to be a floating point number
            }

            u_ij
        }

        pub fn square_well(&mut self, sigma1: f64, sigma2: f64, r: f64, epsilon: f64) -> f64 {
            /*
            // private function
            Return the Square well potential
             */

            let mut u_ij = 0.0;

            if r < sigma1 {
                u_ij = 1000000000000000000000.0; // meant to simulate infinity.. to edit
            } else if sigma1 <= r && r < sigma2 {
                u_ij = -1.0 * epsilon;
            } else {
                u_ij = 0.0;
            }
            u_ij = 2.0;
            u_ij
        }

        fn soft_sphere(
            &mut self,
            sigma1: f64,
            sigma2: f64,
            r: f64,
            epsilon: f64,
            soft_sphere_v: i64,
        ) -> f64 {
            /*
            Return the soft-sphere potential which becomes progressively harder as v increased

             */

            let mut u_ij = 0.0;

            u_ij
        }

        pub fn double_loop(&mut self) -> f64 {
            /*
            ---------------------------------------
            Double loop for lennard-jones potential
            ---------------------------------------

            This code illustrates the calculation of the potential energy
            for a system of Lennard-Jones atoms, using a double loop
            over the atomic indices. The declarations
            at the start are just to remind us of the types

                 */

            //let mut r = vec![vec![0.0, 0.0, 0.0], self.n]; // array of 3d of n length - initialized with zeros
            let mut r: Vec<Vec<f64>> = Vec::with_capacity(self.n as usize);
            let mut rij = vec![0.0, 0.0, 0.0]; // Initial distance array we need
            let mut rng = thread_rng(); // random number generator

            /*
             First of all , fill the values with the random values of points we want.
             We first use the random number generator and generate and collect the values,
             then push those values into the r array

            `*/
            for _ in 0..self.n {
                let random_values: Vec<f64> = (0..3).map(|_| rng.gen::<f64>()).collect();
                r.push(random_values);
            }

            let sigma_sq = self.sigma * self.sigma; // simple sigma squared
            self.pot = 0.0;

            // Loop over the number of particles within the system
            for i in 0..self.n - 1 {
                for j in i + 1..self.n {
                    /*
                    I'm sure I can optimize this part - but for now I will stick to this
                     */
                    let i_usize = i as usize;
                    let j_usize = j as usize;
                    rij[0] = f64::powi(r[i_usize][0] - r[j_usize][0], 2); // x coordinates
                    rij[1] = f64::powi(r[i_usize][1] - r[j_usize][1], 2); // y coordinates
                    rij[2] = f64::powi(r[i_usize][2] - r[j_usize][2], 2); // z coordinates

                    self.rij_sq = rij.iter().sum(); // copute the sum of the three element
                    self.sr2 = sigma_sq / self.rij_sq;
                    self.sr6 = f64::powi(self.sr2, 3);
                    self.sr12 = f64::powi(self.sr6, 3);
                    self.pot = self.pot + self.sr12 - self.sr6
                }
            }
            self.pot = 4.0 * self.epslj * self.pot;
            self.pot
        }

        pub fn site_site_energy_calculation(&mut self, epsilon: f64) -> () {
            /*


            The coordinates r_ia of a site a in molecule i are stored in the elements r(:, i, a)

            For example, if we have two diatomic molecules, then we have r_1a (site a of molecule 1) and
            r_2a (site a of molecule 2). Each molecule is a diatomic molecule (for example, O2). This means that

             */
            let mut rij = vec![0.0, 0.0, 0.0];
            self.na = 1; // creating a diatomic molecule - for example, O2
            let mut r: Vec<Vec<Vec<f64>>> =
                vec![vec![vec![0.0; 3]; self.n as usize]; self.na as usize];

            for a in 0..self.na {
                // loop over
                for b in 0..self.na {
                    for i in 0..self.na {
                        for j in 0..self.na {
                            // lets first try using a square potential - compute the interaction between
                            for k in 0..3 {
                                rij[k] =
                                    r[k][i as usize][a as usize] - r[k][j as usize][b as usize];
                                println!(
                                    "well is {:?}",
                                    self.square_well(self.sigma, self.sigma, rij[k], epsilon)
                                );
                            }
                        }
                    }
                }
            }
        }

        pub fn basic_molecular_dynamics(&mut self) -> () {
            /*
            Running a basic molecular dynamics simulation with the
            above mentioned potentials
             */

            //let positions: Array2<f64> =
            //    Array2::from(vec![vec![0.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]]);
            let mut velocities: Array2<f64> = Array2::zeros((2, 3));

            for step_1 in 0..2 {
                for step_2 in 0..3 {
                    //velocities[[step_1, step_2]] = rng.gen::<f64>() - 0.5;
                }
            }

            // update positions using verlet integration
        }
    }
    //fn site_site_energy(&mut self) {
    //
    //    /*
    //    The coordinates r_{ia} of site a in molecule i are stored
    //    in the elements r(:, i, a) of a rank-3 array; for a system of
    //    diatomic molecules na = 2
    //     */
    //}
}

pub mod general {
    pub struct GeneralStruct {
        // borrwed a slice of an array
        array: [u8; 64], // an array o
        //slice: &array,
        slice: [u8; 64], // an array o
        string_entry: str,
    }

    impl GeneralStruct {
        ///
        ///
        fn print_entry(&self) {
            for entry in &self.slice {
                println!("the entry in the slice is {}", entry);
            }
        }

        //fn split_string(&self) {
        //    for word in &self.string_entry.chars {
        //       println!("{}", word);
        //   }
    }

    fn print_loop(value: &Vec<i32>) {
        let value_clone = value.clone(); // get the cloned value
        for index in &value_clone {
            println!("{} \n", index) // for each value referenced in the index, print out the value index
        }
    }
    // Vec inherits the methods of slices, because we can obtain a slice reference
    // from a vector

    fn print_string(s: String) {
        println!("print_String: {}", s);
    }

    fn print_str(s: &str) {
        println!("print_str: {}", s);
    }
}

pub mod measurement {
    //#[derive(PartialEq, PartialOrd)]
    pub struct Centimeters(f64);

    // Inches, a tuple struct that can be printed
    #[derive(Debug)]
    pub struct Inches(f32);

    // implement method for inches
    //impl Inches {
    //    fn to_centimeter(&self) -> Centimeters {
    //        let &Inches(inches) = self; // get the reference of the inches as self
    // and assign to inches primitive

    //       Centimeters(inches as f64 * 2.54);
    //  }
    //}
}

pub mod satelite {
    /*
    Cubesats are minature satellites that have increasingly increased the
    accessability of space research compared to the conventional satellite.

    A ground station is an intermediary between the operators and the satellites
    themselves. It's listening on the radio, checking on the status of every satellite
    in the constellation and transmitting messages to and fro.

     */

    // Creating the groundstation from where
    // we send information to the satellite

    #[derive(Debug)]
    struct GroundStation;

    #[derive(Debug)]
    struct CubeSat {
        id: u64,
        mailbox: Mailbox, // struct with messages
    }

    #[derive(Debug)]
    struct Mailbox {
        /*
        Mailbox within the satellite to send messages to
        */
        messages: Vec<Message>,
    }

    /*
    Implement messages for the mailbox

    Our cubesat instances die
     */
    impl Mailbox {
        fn post(&mut self, msg: Message) {
            self.messages.push(msg);
        } // requires mutable access to itself and ownership over a message
    }

    // typedef message here
    type Message = String;

    /*
    If we have a large, long standing object such as a global variable,
    it can be somewhat unwieldy to keep this around for every component
    of your program tha needs it.

    In our cubesat case, we don't need to handle much complexity at all.
    Each of our four variables, base, sat_a, sat_b, and sat_c live for the duration

    */
    impl GroundStation {
        /*
        Create methods for the GroundStation to send to
        a mutable CubeSat with a message
         */
        fn send(&self, to: &mut CubeSat, msg: Message) {
            // read-only
            to.mailbox.messages.push(msg); // push a message onto the mailbox
                                           // which is on the Cubsat 'to'
                                           // to.mailbox is
        }
        // Get a sat_id integer as an input, then create a new
        // cubesat with an empty mailbox
        fn connect(&self, sat_id: u64) -> CubeSat {
            CubeSat {
                id: sat_id,
                mailbox: Mailbox { messages: vec![] },
            }
        }
    }

    impl CubeSat {
        fn recv(&mut self) -> () {
            self.mailbox.messages.pop(); // acknowlegement of receiving the mail
        }

        // default input for Cubesat that we will use for testing purposes
        fn default() -> CubeSat {
            // return type atomic parameters
            CubeSat {
                id: 0,
                mailbox: Mailbox {
                    messages: vec![
                        "this".to_string(),
                        "is".to_string(),
                        "a".to_string(),
                        "message".to_string(),
                    ],
                },
            }
        }
    }

    #[derive(Debug)]
    enum StatusMessage {
        Ok,
    }

    fn check_status(sat_id: u64) -> StatusMessage {
        StatusMessage::Ok
    }

    fn check_status_side_effect(sat_id: CubeSat) -> CubeSat {
        /*
        The println! macro takes a reference to 'sat_id' to proint its
        value. Since it is just borrowing a reference to
        'sat_id', ownership is not transferred, and the function still has
        ownership of the 'CubseSat' instance.

         */
        println!("{:?}: {:?}", sat_id, StatusMessage::Ok);
        sat_id // ownership transfer
    }

    // ---

    fn fetch_sat_ids() -> Vec<u64> {
        /*
        Return the satellite ids - simply returns a vector including the satellite ids
        */
        vec![1, 2, 3]
    }

    // example main function we would use
    // fn main() {
    //     let base = GroundStation {};
    //     let mut sat_a = CubeSat {
    //         id: 0,
    //         mailbox: Mailbox { messages: vec![] },
    //     };
    //     println!("t0, {:?}", sat_a);
    //
    //     base.send(&mut sat_a, Message::from("hello there!"));
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    /*
    test the self-consistent field theory part here
     */
    #[test]
    fn test_self_consistent_field() {
        //assert_eq!(
        //    self_consistent_field::atomic_parameters::default().atomic_number,
        //    self_consistent_field::atomic_parameters::default().atomic_number
        //);
        //assert_eq! (
        //        self_consistent_field::atomic_parameters::default()
        //        self_consistent_field::atomic_parameters::default()
        //);
    }

    /*
    test implementation of scf_Vals
     */
    #[test]
    fn test_energy() {
        let scf_entry = self_consistent_field::scf_vals { total_energy: 32.0 };
        let scf_entry_wrong = self_consistent_field::scf_vals { total_energy: 20.0 };
        //assert_eq!(scf_entry, scf_entry_wrong);
    }

    #[test]
    fn test_outer_product() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];

        let result = tensors::outer_product(&a, &b, 0);
        //assert!(result.is::<Vec<Vec<i32>>());
    }

    #[test]
    fn test_atomic_parameters() {
        /*
        Description of what we are testing here
        */
        let mut atom_parameters = self_consistent_field::atomic_parameters {
            ..Default::default()
        };
        atom_parameters.compute_I_values();
        atom_parameters.print_f_values();
        atom_parameters.compute_two_electron_energy();
    }

    #[test]
    fn test_polars() {
        let mut ex = molecular_polars::polars_read_molecular_data_file(
            "/home/sang/Desktop/git/ProgrammingProjects/Project#01/input/benzene.dat",
        );

        let data: Vec<&str> = vec![
            "6    0.000000000000     0.000000000000     0.000000000000",
            "6    0.000000000000     0.000000000000     2.616448463377",
            // Add more strings here
        ];

        //let mut newcontents =
        //    read_lines("/home/sang/Desktop/git/ProgrammingProjects/Project#01/input/benzene.dat");
    }

    // lennard-jones double loop test
    #[test]
    fn test_double_loop() {
        let mut lj_params = LennardJonesSimulations::LJParameters {
            n: 3,
            i: 0,
            j: 1,
            eps: 1.0,
            sigma: 1.0,
            sigma_sq: 0.0,
            pot: 0.0,
            rij_sq: 0.0,
            sr2: 0.0,
            sr6: 0.0,
            sr12: 0.0,
            epslj: 0.0,
        };
        // call the double_loop function
        let result = lj_params.double_loop();

        // assert that the result is as expected
        //assert_eq!(result, expected_result)
    }

    #[test]
    fn test_lennard_jones() {
        let sigma = 1.0;
        let epsilon = 1.0;
        let mut lj_params_new = LennardJonesSimulations::LJParameters {
            n: 3,
            i: 0,
            j: 1,
            eps: 1.0,
            sigma: 1.0,
            sigma_sq: 0.0,
            pot: 0.0,
            rij_sq: 0.0,
            sr2: 0.0,
            sr6: 0.0,
            sr12: 0.0,
            epslj: 0.0,
        };

        // Test case 1 : r = sigma, expected result should be 0
        let r_1 = sigma;
        let result_1 = lj_params_new.lennard_jones(sigma, r_1, epsilon);
        //assert_eq!(result_1, 0.0);
    }

    //#[test]
}
