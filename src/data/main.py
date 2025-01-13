import argparse, sys
import data_collection.labeling as d_labeling
import data_collection.randomizer as d_randomizer
import data_collection.merging as d_merging
import data_analysis.stats as d_stats
import data_analysis.utils as d_utils

from data_collection.generating import OllamaService
from data_processing.processing import Preprocessor

def labeling(input_dataset, output_dataset):
    d_labeling.analyze_sentiments_csv(
        input_file=input_dataset,
        output_file=output_dataset,
    )
    print("Dataset labeled successfully")

def randomizer(input_dataset, output_dataset, sample_size=None):
    try:
        d_randomizer.create_sample(input_dataset, output_dataset, int(sample_size))
        print("Dataset randomization successfully")
    except ValueError:
        d_randomizer.create_sample(input_dataset, output_dataset)
        print("Sample size error, randomized 100000 entry")

def merging(first_input_dataset, second_input_dataset, output_dataset):
    d_utils.Utils.remove_void_or_null(first_input_dataset)
    d_utils.Utils.remove_void_or_null(second_input_dataset)
    d_merging.merge_csv_files(
       file1=first_input_dataset,
       file2=second_input_dataset,
       output_file=output_dataset,
    )
    print("Dataset merged successfully")

def generate(input_dataset, output_dataset, sentiment, mode=None):
    if mode == "adjust":
        d_generate = OllamaService(input_dataset, output_dataset, mode=mode)
        d_generate.process()
    elif mode is None or mode == "generate":
        d_generate = OllamaService(input_dataset, output_dataset, mode="generate", sentiment=sentiment)
        d_generate.process()
    print("Dataset generated successfully")

def generate_stats(input_dataset):
    d_stats.get_stats(input_dataset)

def processing(input_dataset, output_dataset):
    d_preprocessor = Preprocessor()
    d_preprocessor.load_dataset(input_dataset)
    d_preprocessor.preprocess_text(output_dataset)

def main():

    # Main parser
    parser = argparse.ArgumentParser(
        description="A tool to perform operations on datasets: collection, processing, or statistics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for data_collection
    data_collection_parser = subparsers.add_parser(
        "data_collection", help="Perform data collection operations like labeling, randomizer, generating, or merging."
    )
    data_collection_parser.add_argument(
        "--operation",
        type=str,
        choices=["labeling", "randomizer", "generating", "merging"],
        help="The data collection operation to perform.",
        required=True
    )
    data_collection_parser.add_argument(
        "collection_arg1",
        type=str,
        nargs="?",
        help="The first input dataset or operation parameter."
    )
    data_collection_parser.add_argument(
        "collection_arg2",
        type=str,
        nargs="?",
        help="The second input dataset for operations like 'labeling' and 'generating'."
    )
    data_collection_parser.add_argument(
        "collection_arg3",
        type=str,
        nargs="?",
        help="The output dataset for operations like 'labeling' and 'generating'."
    )
    data_collection_parser.add_argument(
        "--mode",
        type=str,
        choices=["adjust", "generate"],
        help="The mode for dataset generation (default: 'generate')."
    )
    data_collection_parser.add_argument(
        "--sentiment",
        type=str,
        choices=["neutral", "positive", "negative"],
        help="The sentiment for dataset generation (default: 'negative')."
    )

    # Subparser for data_processing
    data_processing_parser = subparsers.add_parser(
        "data_processing", help="Perform data processing operations like cleaning or transformations."
    )
    data_processing_parser.add_argument(
        "processing_arg1",
        type=str,
        help="The input dataset to process."
    )
    data_processing_parser.add_argument(
        "processing_arg2",
        type=str,
        help="The processed output dataset."
    )

    # Subparser for data_stats
    data_stats_parser = subparsers.add_parser(
        "data_stats", help="Generate statistics on datasets."
    )
    data_stats_parser.add_argument(
        "stats_arg1",
        type=str,
        help="The dataset to calculate statistics on."
    )

    args = parser.parse_args()

    try:
        # Handle the "data_collection" command
        if args.command == "data_collection":
            if args.operation == "labeling":
                if args.sentiment or args.mode:
                    print("Error: --sentiment and --mode are not allowed with the 'labeling' operation.")
                    sys.exit(1)
                if not args.collection_arg1 or not args.collection_arg2:
                    print("Error: 'labeling' requires two datasets (input and output).")
                    sys.exit(1)
                labeling(args.collection_arg1, args.collection_arg2)

            elif args.operation == "randomizer":
                if args.sentiment or args.mode:
                    print("Error: --sentiment and --mode are not allowed with the 'randomizer' operation.")
                    sys.exit(1)
                if not args.collection_arg1 or not args.collection_arg2:
                    print("Error: 'randomizer' requires two datasets (input and output).")
                    sys.exit(1)
                if args.collection_arg3:
                    randomizer(args.collection_arg1, args.collection_arg2, args.collection_arg3)
                else:
                    randomizer(args.collection_arg1, args.collection_arg2)

            elif args.operation == "merging":
                if args.sentiment or args.mode:
                    print("Error: --sentiment and --mode are not allowed with the 'merging' operation.")
                    sys.exit(1)
                if not args.collection_arg1 or not args.collection_arg2 or not args.collection_arg3:
                    print("Error: 'merging' requires three datasets (two input and one output).")
                    sys.exit(1)
                merging(args.collection_arg1, args.collection_arg2, args.collection_arg3)

            elif args.operation == "generating":
                if not args.collection_arg1 or not args.collection_arg2 or not args.sentiment:
                    print("Error: 'generating' requires two datasets (input and output) and a sentiment.")
                    sys.exit(1)
                if args.mode:
                    generate(args.collection_arg1, args.collection_arg2, args.sentiment, args.mode)
                else:
                    generate(args.collection_arg1, args.collection_arg2, args.sentiment)

        # Handle the "data_processing" command
        elif args.command == "data_processing":
            if not args.processing_arg1 or not args.processing_arg2:
                print("Error: 'data_processing' requires an input dataset and an output dataset.")
                sys.exit(1)
            processing(args.processing_arg1, args.processing_arg2)

        # Handle the "data_stats" command
        elif args.command == "data_stats":
            if not args.stats_arg1:
                print("Error: 'data_stats' requires a dataset.")
                sys.exit(1)
            generate_stats(args.stats_arg1)

        else:
            print("Error: Invalid command or missing arguments.")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
else: pass