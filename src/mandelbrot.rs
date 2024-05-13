pub mod mandelbrot {
    use num::complex::Complex;
    use rand::{random, Rng};
    use std::cmp::Ordering;
    use std::io;
    use std::thread;
    use std::time::Duration;

    // Libaries necessary to run all functions within
    // this module

    pub fn simulated_expensive_calculation(intensity: u32) -> u32 {
        println!("Calculating slowly");
        thread::sleep(Duration::from_secs(2));
        intensity
    }

    // We require an intensity number from the user, which is specified when they request
    // a workout to indicate whether they want a low-intensity workout or high intensity workout

    pub fn generate_workout(intensity: u32, random_number: u32) {
        /*
         */
        if intensity < 25 {
            println!("{} exercises", simulated_expensive_calculation(intensity));
        } else {
            if random_number == 3 {
                println!("take a break today! Remember to stay hydrated");
            } else {
                println!(
                    "Today, run for {} minutes!",
                    simulated_expensive_calculation(intensity)
                );
            }
        }
    }

    pub fn match_function() {
        /*
        Packing up the previous functions into here
        from the guess, see if the number fits
         */

        let secret_number = rand::thread_rng().gen_range(1..101);

        println!("The secret number is: {}", secret_number);

        loop {
            println!("Please input your guess");
            let mut guess = String::new(); // Rust has a strong, static type system.
            io::stdin()
                .read_line(&mut guess)
                .expect("Failed to read line"); // handling potential failure with the result type

            let guess: u32 = match guess.trim().parse() {
                Ok(num) => num,
                Err(_) => continue,
            };

            match guess.cmp(&secret_number) {
                Ordering::Less => println!("Too Small"),
                Ordering::Greater => println!("Too big"),
                //Ordering::Equal => println!("You win!"),
                Ordering::Equal => {
                    println!("You win!");
                    break;
                }
            }
        }
    }

    pub fn calculate_mandelbrot(
        max_iters: usize,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        width: usize,
        height: usize,
    ) -> Vec<Vec<usize>> {
        let mut all_rows: Vec<Vec<usize>> = Vec::with_capacity(width); // Create container
        for img_y in 0..height {
            // Loop between 0 and height value
            let mut row: Vec<usize> = Vec::with_capacity(height); // Define new
                                                                  // Vector with every looed height
            for img_x in 0..width {
                let cx = x_min + (x_max - x_min) * (img_x as f64 / width as f64);
                let cy = y_min + (y_max - y_min) * (img_y as f64 / height as f64);
                let escaped_at = mandelbrot_at_point(cx, cy, max_iters);
                row.push(escaped_at);
            }
            all_rows.push(row);
        }
        return all_rows;
    }

    fn mandelbrot_at_point(cx: f64, cy: f64, max_iters: usize) -> usize {
        ///create mandelbrot plot
        let mut z = Complex { re: 0.0, im: 0.0 };
        let c = Complex::new(cx, cy);

        for i in 0..=max_iters {
            if z.norm() > 2.0 {
                // norm computes the absolute value of the complex number
                return i;
            }
            z = z * z + c;
        }

        return max_iters;
    }

    fn render_mandelbrot(escape_vals: Vec<Vec<usize>>) {
        for row in escape_vals {
            let mut line = String::with_capacity(row.len());
            for column in row {
                let val = match column {
                    0..=2 => ' ',
                    2..=5 => '.',
                    5..=10 => '.',
                    11..=30 => '*',
                    30..=100 => '+',
                    100..=200 => 'x',
                    200..=400 => '$',
                    400..=700 => '#',
                    _ => '%',
                };
                line.push(val);
            }
            println!("{}", line);
        }
    }
}
