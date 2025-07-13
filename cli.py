import click

def load_and_preprocess_data():
    """Load and preprocess the data."""
    from src.data.loader import load_data
    from src.utils.plotting import plot_data
    from src.features.indicators import calculate_technical_indicators
    from src.data.preprocess import normalize, denormalize, build_train_data, split_data

    df = load_data()
    df = calculate_technical_indicators(df)
    df = df.dropna()
    
    # Normalize the data
    normalized_df = normalize(df)
    
    return normalized_df, df

def train_test_split(normalized_df, past_days, future_days, test_size):
    """Split the data into training and testing sets."""
    from src.data.preprocess import build_train_data, split_data
    x_train, y_train = build_train_data(normalized_df, past_days=past_days, future_days=future_days)
    x_train, y_train, x_test, y_test = split_data(x_train, y_train, test_size=test_size)
    return x_train, y_train, x_test, y_test

@click.group()
def cli():
    """LSTM Bitcoin Price Prediction CLI - A command line interface for managing Bitcoin price prediction tasks."""
    from src.utils.reproducibility import set_seed

    set_seed(42)  # Set seed for reproducibility

@cli.command()
def clean():
    """Clean the raw data for analysis."""
    from src.data.cleaning import clean

    clean()

@cli.command()
@click.option('--past-days', default=20, help='Number of past days to use for prediction')
@click.option('--future-days', default=1, help='Number of future days to predict')
@click.option('--test-size', default=0.2, help='Proportion of data to use for testing')
@click.option('--epochs', default=20, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Batch size for training')
def train(past_days, future_days, test_size, epochs, batch_size):
    """Train the LSTM model."""
    from src.models.lstm import train_model

    normalized_df, _ = load_and_preprocess_data()
    x_train, y_train, _, _ = train_test_split(normalized_df, past_days=past_days, future_days=future_days, test_size=test_size)

    train_model(x_train, y_train, epochs=epochs, batch_size=batch_size)
    print("Model trained and saved to models/lstm_paper.keras")

@cli.command()
@click.option('--model-path', default="models/lstm_paper.keras", help='Path to the trained model')
@click.option('--past-days', default=20, help='Number of past days used in training')
@click.option('--future-days', default=1, help='Number of future days to predict')
@click.option('--test-size', default=0.2, help='Proportion of data used for testing')
@click.option('--calibrate/--no-calibrate', default=True, help='Apply STBC calibration')
@click.option('--optimize/--no-optimize', default=True, help='Optimize STBC parameters with SSGA')
@click.option('--pop-size', default=100, help='Population size for SSGA')
@click.option('--generations', default=15, help='Number of generations for SSGA')
@click.option('--save/--no-save', default=True, help='Save results to disk')
def predict(model_path, past_days, future_days, test_size, calibrate, optimize, pop_size, generations, save): 
    """Predict using the trained LSTM model."""
    import numpy as np

    from src.models.lstm import predict_model
    from src.metrics import evaluate_mse
    from src.data.saver import save_results
    from src.data.preprocess import denormalize

    normalized_df, original_df = load_and_preprocess_data()
    _, _, x_test, _ = train_test_split(normalized_df, past_days=past_days, future_days=future_days, test_size=test_size)

    # Predict using the trained model
    predictions = predict_model(model_path, x_test)
    
    # Denormalize predictions
    test_start_idx = original_df.shape[0] - predictions.shape[0]
    original_close = original_df['close'].copy()
    original_close_test = original_close[test_start_idx:]
    predictions = denormalize(predictions, original_close_test)

    # seperate the data into train and test
    original_train = original_df[:test_start_idx]
    original_test = original_df[test_start_idx:].copy()

    # add the predictions to the test data
    original_test["Predictions"] = predictions

    # drop the columns that are not needed
    original_test = original_test.drop(["open", "high", "low", "k_9", "d_9", "ma_6", "ma_9", "ma_12", "bias_6", "bias_3", "bias_3_minus_bias_6", "rsi_6", "williams_r_12", "momentum_6", "momentum_6_ma", "macd_9", "macd_signal_9", "macd_diff"], axis=1)

    original_test["diff"] = original_test["Predictions"] - original_test["close"]
    original_test["diff_percentage"] = original_test["diff"] / original_test["close"]

    if save:
        save_results(original_test, "Predictions")

    print("MSE without STBC:" + str(evaluate_mse(original_test)))
    print("Average prediction error without STBC:" + str(np.mean(np.abs(original_test["diff_percentage"]))))

    if calibrate:
        from src.bias.stbc import STBC

        stbc = STBC(original_test, 0.05)
        stbc.calibrate()
        print("MSE after optimizing STBC:" + str(stbc.evaluate_mse()))
        print("Average prediction error after optimizing STBC:" + str(stbc.evaluate_accuracy()))

        if optimize:
            from src.optim.ssga import SSGA

            ssga = SSGA(
                df=original_test,
                pop_size=pop_size,
                num_generations=generations,
                mutation_rate=0.01,
                crossover_rate=1,
                tournament_size=5,
                random_seed=42
            )

            print("STBC Optimizing using SSGA...")
            best_individual, best_fitness = ssga()
            print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")

            stbc = STBC(original_test, best_individual)
            stbc.calibrate()
            print("MSE after optimizing STBC:" + str(stbc.evaluate_mse()))
            print("Average prediction error after optimizing STBC:" + str(stbc.evaluate_accuracy()))

            if save:
                save_results(stbc.df, "STBC Predictions")

if __name__ == "__main__":
    cli()