import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Any
import pytorch_forecasting
from dataclasses import dataclass

@dataclass
class BenchmarkData:
    index: Optional[Any] = None
    scaler: Optional[Any] = None
    description: Optional[str] = None

def plotPred(model, rawPredictions, idx=0, ax=None, benchmarkData: Optional[BenchmarkData] = None):
    # Validate inputs
    if not hasattr(model, 'to_prediction') or not hasattr(model, 'to_quantiles'):
        raise ValueError("Model must have 'to_prediction' and 'to_quantiles' methods.")
    if not hasattr(rawPredictions, 'x') or not hasattr(rawPredictions, 'output'):
        raise ValueError("rawPredictions must have 'x' and 'output' attributes.")

    try:
        # Set up color cycle for plotting
        propCycle = iter(plt.rcParams["axes.prop_cycle"])
        obsColor = next(propCycle)["color"]
        predColor = next(propCycle)["color"]

        # Extract data from raw predictions
        encoderTargets = pytorch_forecasting.utils.to_list(rawPredictions.x["encoder_target"])
        decoderTargets = pytorch_forecasting.utils.to_list(rawPredictions.x["decoder_target"])
        encoderCont = pytorch_forecasting.utils.to_list(rawPredictions.x["encoder_cont"])
        decoderCont = pytorch_forecasting.utils.to_list(rawPredictions.x["decoder_cont"])
        yHats = pytorch_forecasting.utils.to_list(model.to_prediction(rawPredictions.output))
        yQuantiles = pytorch_forecasting.utils.to_list(model.to_quantiles(rawPredictions.output))

        for yHat, yQuantile, encoderTarget, decoderTarget, encCont, decCont in zip(yHats, yQuantiles, encoderTargets, decoderTargets, encoderCont, decoderCont):
            yAll = torch.cat([encoderTarget[idx], decoderTarget[idx]])
            maxEncoderLength = rawPredictions.x["encoder_lengths"].max()
            y = torch.cat([
                yAll[:rawPredictions.x["encoder_lengths"][idx]],
                yAll[maxEncoderLength:maxEncoderLength + rawPredictions.x["decoder_lengths"][idx]]
            ])

            # Move predictions to CPU
            yHat = yHat.detach().cpu()[idx, :rawPredictions.x["decoder_lengths"][idx]]
            yQuantile = yQuantile.detach().cpu()[idx, :rawPredictions.x["decoder_lengths"][idx]]
            y = y.detach().cpu()

            nPred = yHat.shape[0]
            xObs = np.arange(-(y.shape[0] - nPred), 0)
            xPred = np.arange(nPred)

            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()

            smapeScore = smape(y[-nPred:].numpy(), yHat.numpy())
            ax.set_title(f"{type(model).__name__} {np.round(100 - smapeScore)}%")

            # Plot observed history
            plotter = ax.plot if len(xObs) > 1 else ax.scatter
            if len(xObs) > 0:
                plotter(xObs, y[:-nPred], label="observed", c=obsColor)

            # Plot benchmark if available
            if benchmarkData is not None:
                try:
                    benchmarkObs = benchmarkData.scaler.inverse_transform(encCont[idx, :, benchmarkData.index].reshape(-1, 1))
                    benchmarkPred = benchmarkData.scaler.inverse_transform(decCont[idx, :, benchmarkData.index].reshape(-1, 1))
                    plotter(xObs, benchmarkObs, label=benchmarkData.description, c="r")
                    plotter(xPred, benchmarkPred, label=None, c="r")
                except Exception as e:
                    logging.error(f"Error plotting benchmark data: {e}")

            # Plot observed prediction
            plotter(xPred, y[-nPred:], label=None, c=obsColor)

            # Plot predicted values
            plotter(xPred, yHat, label="predicted", c=predColor)
            plotter(xPred, yQuantile[:, yQuantile.shape[1] // 2], c=predColor, alpha=0.15)

            # Plot prediction quantiles
            for i in range(yQuantile.shape[1] // 2):
                if len(xPred) > 1:
                    ax.fill_between(xPred, yQuantile[:, i], yQuantile[:, -i - 1], alpha=0.15, fc=predColor)
                else:
                    quantiles = torch.tensor([[yQuantile[0, i]], [yQuantile[0, -i - 1]]])
                    ax.errorbar(
                        xPred,
                        y[[-nPred]],
                        yerr=quantiles - y[-nPred],
                        c=predColor,
                        capsize=1.0,
                    )
    except Exception as e:
        logging.error(f"Error in plotPred: {e}")
        raise

# Example usage:
# model = ... # Your model
# rawPredictions = ... # Your raw predictions
# benchmarkData = BenchmarkData(index=[0], scaler=someScaler, description="Benchmark")
# plotPred(model, rawPredictions, idx=0, ax=None, benchmarkData=benchmarkData)
