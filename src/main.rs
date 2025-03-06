use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;

struct InMemDataset {
    records: Vec<Vec<String>>,
}

impl InMemDataset {
    fn from_csv(path: &str, reader_builder: ReaderBuilder) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut rdr = reader_builder.from_reader(file);
        let mut records = Vec::new();

        let headers = rdr.headers()?;
        println!("Column names: {:?}", headers);
        for (i, result) in rdr.records().enumerate() {
            if i >= 1 {
                break;
            }
            let record = result?;
            println!("{:?}", record);
        }
        //TRIP_ID KEEP
        //CALL_TYPE REMOVE
        //ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, DAYTYPE REMOVE
        //TIMESTAMP KEEP
        //MISSING_DATA REMOVE TRUE RECORDS?

        for result in rdr.records() {
            let record = result?;
            records.push(record.iter().map(|s| s.to_string()).collect());
        }
        
        Ok(Self { records })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = ReaderBuilder::new();
    builder.delimiter(b','); // modify in place; don't reassign the result

    // Now, builder still owns a ReaderBuilder that can be passed by value.
    let dataset = InMemDataset::from_csv(
        r"C:\Users\dgl\Documents\ruststuff\burn-lstm\train.csv",
        builder,
    )?;
    
    println!("Loaded {} records", dataset.records.len());
    Ok(())
}