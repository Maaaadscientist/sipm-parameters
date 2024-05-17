import os
import argparse
import pandas as pd


def read_csv_file(csv_file):
    if os.path.isdir(csv_file):
        all_data = []
        for filename in os.listdir(csv_file):
            if filename.endswith(".csv"):
                file_path = os.path.join(csv_file, filename)
                data = pd.read_csv(file_path)
                all_data.append(data)
        if len(all_data) == 0:
            raise ValueError(f"The directory {csv_file} contains no CSV files.")
        df = pd.concat(all_data, ignore_index=True)
    else:
        if csv_file.endswith(".csv"):
            df = pd.read_csv(csv_file)
        else:
            raise ValueError("Provided file is not a valid CSV file.")
    return df


def fill_empty_elements(csv_str):
    while ',,' in csv_str:
        csv_str = csv_str.replace(',,', ',0,')
    
    csv_str = "\n".join([line if not line.endswith(",") else line + "0" for line in csv_str.split("\n")])
    return csv_str


def main(args):
    # Split the drop_columns string into a list
    drop_columns = args.drop_columns.split(",")
    
    # Initialize the merged DataFrame with the first CSV file or directory
    merged_df = read_csv_file(args.csv_files[0])
    
    for csv_file in args.csv_files[1:]:
        df = read_csv_file(csv_file).drop(columns=drop_columns, errors='ignore')
        
        # Rename columns if rename_columns argument is provided
        if args.rename_columns:
            rename_dict = {}
            for rename_pair in args.rename_columns:
                old_name, new_name = rename_pair.split(",")
                rename_dict[old_name] = new_name
            df = df.rename(columns=rename_dict)
        
        # Merge the current DataFrame with the merged DataFrame
        merged_df = pd.merge(merged_df, df, on=args.key_columns.split(","), how='outer')

    csv_str = merged_df.to_csv(index=False)
    modified_csv_str = fill_empty_elements(csv_str)

    output_path = os.path.abspath(args.output_path)
    output_dir = os.path.dirname(output_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, 'w') as f:
        f.write(modified_csv_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple CSV files or directories containing CSV files.')
    parser.add_argument('csv_files', nargs='+', help='CSV files or directories to merge')
    parser.add_argument('output_path', help='Path for the merged CSV output')
    parser.add_argument('-k', '--key_columns', default='run,pos,ch,vol',
                        help='Columns to use as keys for merging. Default is "run,pos,ch,vol".')
    parser.add_argument('-d', '--drop_columns', default='',
                        help='Columns to drop before merging. Provide comma-separated column names.')
    parser.add_argument('-n', '--rename_columns', action='append', default=[],
                        help='Columns to rename. Provide pairs like old_name,new_name. Repeat -n for each pair.')

    args = parser.parse_args()
    main(args)

