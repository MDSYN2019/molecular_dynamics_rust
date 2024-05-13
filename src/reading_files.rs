#[derive(Debug)]
struct reading_coordinate_files {
    filename: String,
    atomic_data: DataFrame,
    //stored_atomic_data: Vec<str>::new(),
}
// implementing the sumamry trait for reading_coordinate_files struct
impl Summary for reading_coordinate_files {
    fn summarize(&self) -> String {
        format!("are the filename and atomic_data respectively") // todo
    }
}

// Taking the edited benzene file example
impl reading_coordinate_files {
    fn new(name: &str) -> reading_coordinate_files {
        /*
        Basically making a constructor for this struct.
        */
        reading_coordinate_files {
            filename: String::from(
                "/home/sang/Desktop/git/ProgrammingProjects/Project#01/input/benzene_new.dat",
            ),
            atomic_data: DataFrame::default(),
        }
    }
}

fn read_lines(filename: &str) -> Vec<String> {
    fs::read_to_string(filename)
        .unwrap()
        .lines()
        .map(String::from)
        .collect()
}

fn example() -> PolarsResult<DataFrame> {
    CsvReader::from_path(
        "/home/sang/Desktop/git/ProgrammingProjects/Project#01/input/benzene_new.dat",
    )?
    .has_header(true)
    .finish()
}
