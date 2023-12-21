###############################
### DO NOT CHANGE THIS FILE ###
###############################

# Helper functions to visualize sample regression data

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt

N_SAMPLES = 700
PERCENT_TRAIN = 0.8


class Plotter:
    def __init__(
        self,
        regularization,
        poly_degree,
        student_version,
        eo_params,
        print_images=False,
    ):
        self.reg = regularization
        self.POLY_DEGREE = poly_degree
        self.print_images = print_images
        self.STUDENT_VERSION = student_version

        (
            self.EO_TEXT,
            self.EO_FONT,
            self.EO_COLOR,
            self.EO_ALPHA,
            self.EO_SIZE,
            self.EO_ROT,
        ) = eo_params

        self.rng = np.random.RandomState(seed=10)

        self.camera = dict(eye=dict(x=1, y=-1.90, z=0.8), up=dict(x=0, y=0, z=1))

    def create_data(self):
        rng = self.rng

        # Simulating a regression dataset with polynomial features.
        true_weight = rng.rand(self.POLY_DEGREE**2 + 2, 1)
        x_feature1 = np.linspace(-5, 5, N_SAMPLES)
        x_feature2 = np.linspace(-3, 3, N_SAMPLES)
        x_all = np.stack((x_feature1, x_feature2), axis=1)

        reg = self.reg
        x_all_feat = reg.construct_polynomial_feats(x_all, self.POLY_DEGREE)
        x_cart_flat = []
        for i in range(x_all_feat.shape[0]):
            point = x_all_feat[i]
            x1 = point[:, 0]
            x2 = point[:, 1]
            x1_end = x1[-1]
            x2_end = x2[-1]
            x1 = x1[:-1]
            x2 = x2[:-1]
            x3 = np.asarray([[m * n for m in x1] for n in x2])

            x3_flat = list(np.reshape(x3, (x3.shape[0] ** 2)))
            x3_flat.append(x1_end)
            x3_flat.append(x2_end)
            x3_flat = np.asarray(x3_flat)
            x_cart_flat.append(x3_flat)

        x_cart_flat = np.asarray(x_cart_flat)
        x_cart_flat = (x_cart_flat - np.mean(x_cart_flat)) / np.std(
            x_cart_flat
        )  # Normalize
        x_all_feat = np.copy(x_cart_flat)

        p = np.reshape(np.dot(x_cart_flat, true_weight), (N_SAMPLES,))
        # We must add noise to data, else the data will look unrealistically perfect.
        y_noise = rng.randn(x_all_feat.shape[0], 1)
        y_all = np.dot(x_cart_flat, true_weight) + y_noise
        print(
            "x_all: ",
            x_all.shape[0],
            " (rows/samples) ",
            x_all.shape[1],
            " (columns/features)",
            sep="",
        )
        print(
            "y_all: ",
            y_all.shape[0],
            " (rows/samples) ",
            y_all.shape[1],
            " (columns/features)",
            sep="",
        )

        return x_all, y_all, p, x_all_feat, x_cart_flat

    def split_data(self, x_all, y_all):
        rng = self.rng

        # Generate Train/Test Split
        all_indices = rng.permutation(N_SAMPLES)  # Random indicies
        train_indices = all_indices[: round(N_SAMPLES * PERCENT_TRAIN)]  # 80% Training
        test_indices = all_indices[round(N_SAMPLES * PERCENT_TRAIN) :]  # 20% Testing

        xtrain = x_all[train_indices]
        ytrain = y_all[train_indices]
        xtest = x_all[test_indices]
        ytest = y_all[test_indices]

        return xtrain, ytrain, xtest, ytest, train_indices, test_indices

    def plot_all_data(self, x_all, y_all, p):
        rndr_type = "jupyterlab+png"
        pio.renderers.default = rndr_type
        # Render types : 'browser', 'png', 'plotly_mimetype', 'jupyterlab', pdf

        df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "y": np.squeeze(y_all),
                "best_fit": np.squeeze(p),
            }
        )

        # Initialize the figure
        fig = go.Figure()

        # Add scatter points to the figure with a legend name
        fig.add_scatter3d(
            x=df["feature1"],
            y=df["feature2"],
            z=df["y"],
            mode="markers",
            marker=dict(color="blue", opacity=0.12),
            name="Data Points",
        )

        # Add the line of best fit to the figure
        fig.add_scatter3d(
            x=df["feature1"],
            y=df["feature2"],
            z=df["best_fit"],
            mode="lines",
            line=dict(color="red", width=7),
            name="Line of Best Fit",
        )

        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=self.EO_SIZE, color="gray"),
                opacity=0.5,
                align="center",
                textangle=-self.EO_ROT,
            )

        # Update the layout
        fig.update_layout(
            title="All Simulated Datapoints",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=self.camera,
            ),
            height=700,
            width=1000,
            autosize=True,
        )
        # Show the figure
        config = {"scrollZoom": True}
        fig.show(config=config)

        if self.print_images:
            fig.write_image("outputs/Data_Regression_Truth.png")
            img = mpimg.imread("outputs/Data_Regression_Truth.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_split_data(self, xtrain, xtest, ytrain, ytest):
        # Initialize the Plotly figure
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)

        # Create a DataFrame
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )
        all_data = pd.concat([train_df, test_df])

        # Initialize the Plotly figure
        fig = go.Figure()

        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )

        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )

        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=self.EO_SIZE, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )

        fig.update_layout(
            title="Data Set Split",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=self.camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )

        # Show the figure
        fig.show()
        # Create and print static images
        if self.print_images:
            fig.write_image("outputs/Samples_Regression_Train-Test.png")
            img = mpimg.imread("outputs/Samples_Regression_Train-Test.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_linear_closed(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        # Initialize the Plotly figure

        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        # Create a DataFrame
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )

        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )

        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )

        all_data = pd.concat([train_df, test_df])

        # Initialize the Plotly figure
        fig = go.Figure()

        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )

        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="red", width=7),
            name="Trendline",
        )

        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=80, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )
        fig.update_layout(
            title="Linear (Closed)",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=self.camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )
        # Show the figure
        fig.show()
        if self.print_images:
            fig.write_image("outputs/Linear_Fit_Closed.png")
            img = mpimg.imread("outputs/Linear_Fit_Closed.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_linear_gd(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        # Initialize the Plotly figure

        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        # Create a DataFrame
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )

        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )

        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )

        all_data = pd.concat([train_df, test_df])

        # Initialize the Plotly figure
        fig = go.Figure()

        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )

        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="red", width=7),
            name="Trendline",
        )

        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=80, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )
        fig.update_layout(
            title="Linear (GD)",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=self.camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )
        # Show the figure
        fig.show()
        if self.print_images:
            fig.write_image("outputs/Linear_Fit_GD.png")
            img = mpimg.imread("outputs/Linear_Fit_GD.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_linear_gd_tuninglr(
        self, xtrain, xtest, ytrain, ytest, x_all, x_all_feat, learning_rates, weights
    ):
        reg = self.reg

        # Initialize the Plotly figure
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        # Create a DataFrame
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )

        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )

        all_data = pd.concat([train_df, test_df])
        # Initialize the Plotly figure
        fig = go.Figure()
        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )
        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )
        # Add fitting line
        colors = ["green", "blue", "pink"]
        for ii in range(len(learning_rates)):
            y_pred = reg.predict(x_all_feat, weights[ii])
            y_pred = np.reshape(y_pred, (y_pred.size,))

            pred_df = pd.DataFrame(
                {
                    "feature1": x_all[:, 0],
                    "feature2": x_all[:, 1],
                    "Trendline": np.squeeze(y_pred),
                }
            )
            fig.add_scatter3d(
                x=pred_df["feature1"],
                y=pred_df["feature2"],
                z=pred_df["Trendline"],
                mode="lines",
                line=dict(color=colors[ii], width=7),
                name="Trendline LR=" + str(learning_rates[ii]),
            )
        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=80, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )
        fig.update_layout(
            title="Tuning Linear (GD)",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=self.camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )
        # Show the figure
        fig.show()
        if self.print_images:
            fig.write_image("outputs/Linear_Fit_GD_Learning_Rates.png")
            img = mpimg.imread("outputs/Linear_Fit_GD_Learning_Rates.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_linear_closed_10samples(self, x_all, y_all_noisy, sub_train, y_pred):
        # Create a DataFrame
        samples_df = pd.DataFrame(
            {
                "feature1": x_all[sub_train, 0],
                "feature2": x_all[sub_train, 1],
                "y": np.squeeze(y_all_noisy[sub_train]),
                "label": "Samples",
            }
        )
        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.reshape(y_pred, (N_SAMPLES,)),
            }
        )

        # Initialize the Plotly figure
        fig = go.Figure()
        # Add training data
        fig.add_scatter3d(
            x=samples_df["feature1"],
            y=samples_df["feature2"],
            z=samples_df["y"],
            mode="markers",
            marker=dict(color="red", size=10, opacity=0.75),
            name="Samples",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="blue", width=7),
            name="Trendline",
        )
        fig.update_traces(
            marker=dict(size=8, symbol="x", line=dict(width=2, color="red")),
            selector=dict(mode="markers"),
        )
        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=80, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )
        camera = dict(eye=dict(x=0.75, y=-1.75, z=1.5), up=dict(x=0, y=0, z=1))
        fig.update_layout(
            title="Linear Regression (Closed)",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )
        # Show the figure
        fig.show()
        if self.print_images:
            fig.write_image("outputs/Linear_Regression_Closed_10_Samples.png")
            img = mpimg.imread("outputs/Linear_Regression_Closed_10_Samples.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_ridge_closed_10samples(self, x_all, y_all_noisy, sub_train, y_pred):
        # Create a DataFrame
        samples_df = pd.DataFrame(
            {
                "feature1": x_all[sub_train, 0],
                "feature2": x_all[sub_train, 1],
                "y": np.squeeze(y_all_noisy[sub_train]),
                "label": "Samples",
            }
        )
        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )
        # Initialize the Plotly figure
        fig = go.Figure()
        # Add training data
        fig.add_scatter3d(
            x=samples_df["feature1"],
            y=samples_df["feature2"],
            z=samples_df["y"],
            mode="markers",
            marker=dict(color="red", size=10, opacity=0.75),
            name="Samples",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="blue", width=7),
            name="Trendline",
        )
        fig.update_traces(
            marker=dict(size=8, symbol="x", line=dict(width=2, color="red")),
            selector=dict(mode="markers"),
        )
        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=80, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )
        camera = dict(eye=dict(x=0.75, y=-1.75, z=1.5), up=dict(x=0, y=0, z=1))
        fig.update_layout(
            title="Ridge Regression (Closed)",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )
        # Show the figure
        fig.show()
        if self.print_images:
            fig.write_image("outputs/Ridge_Regression_Closed_10_Samples.png")
            img = mpimg.imread("outputs/Ridge_Regression_Closed_10_Samples.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_ridge_gd_10samples(self, x_all, y_all_noisy, sub_train, y_pred):
        # Create a DataFrame
        samples_df = pd.DataFrame(
            {
                "feature1": x_all[sub_train, 0],
                "feature2": x_all[sub_train, 1],
                "y": np.squeeze(y_all_noisy[sub_train]),
                "label": "Samples",
            }
        )
        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )
        # all_data = pd.concat([train_df, test_df])
        # Initialize the Plotly figure
        fig = go.Figure()
        # Add training data
        fig.add_scatter3d(
            x=samples_df["feature1"],
            y=samples_df["feature2"],
            z=samples_df["y"],
            mode="markers",
            marker=dict(color="red", size=10, opacity=0.75),
            name="Samples",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="blue", width=7),
            name="Trendline",
        )
        fig.update_traces(
            marker=dict(size=8, symbol="x", line=dict(width=2, color="red")),
            selector=dict(mode="markers"),
        )
        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=80, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )
        camera = dict(eye=dict(x=0.75, y=-1.75, z=1.5), up=dict(x=0, y=0, z=1))
        fig.update_layout(
            title="Ridge Regression (GD)",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )
        # Show the figure
        fig.show()
        if self.print_images:
            fig.write_image("outputs/Ridge_Regression_GD_10_Samples.png")
            img = mpimg.imread("outputs/Ridge_Regression_GD_10_Samples.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

    def plot_ridge_sgd_10samples(self, x_all, y_all_noisy, sub_train, y_pred):
        # Create a DataFrame
        samples_df = pd.DataFrame(
            {
                "feature1": x_all[sub_train, 0],
                "feature2": x_all[sub_train, 1],
                "y": np.squeeze(y_all_noisy[sub_train]),
                "label": "Samples",
            }
        )
        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )
        # Initialize the Plotly figure
        fig = go.Figure()
        # Add training data
        fig.add_scatter3d(
            x=samples_df["feature1"],
            y=samples_df["feature2"],
            z=samples_df["y"],
            mode="markers",
            marker=dict(color="red", size=10, opacity=0.75),
            name="Samples",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="blue", width=7),
            name="Trendline",
        )
        fig.update_traces(
            marker=dict(size=8, symbol="x", line=dict(width=2, color="red")),
            selector=dict(mode="markers"),
        )
        # Add watermark, if not the student version
        if not self.STUDENT_VERSION:
            fig.add_annotation(
                text=self.EO_TEXT,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=80, color="gray"),  # Adjust size as needed
                opacity=self.EO_ALPHA,
                align="center",
                textangle=-self.EO_ROT,
            )
        camera = dict(eye=dict(x=0.75, y=-1.75, z=1.5), up=dict(x=0, y=0, z=1))
        fig.update_layout(
            title="Ridge Regression (SGD)",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=camera,
            ),
            autosize=True,
            width=800,
            height=700,
        )
        # Show the figure
        fig.show()
        if self.print_images:
            fig.write_image("outputs/Ridge_Regression_SGD_10_Samples.png")
            img = mpimg.imread("outputs/Ridge_Regression_SGD_10_Samples.png")
            plt.imshow(img)
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()
