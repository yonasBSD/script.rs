#![cfg(all(not(feature = "no_plots"), feature = "plots"))]
//! Rhai scripting engine integration
//!
//! Provides a sandboxed scripting environment for complex task logic.
//! Rhai was chosen for its fast startup time, Rust-native integration,
//! and familiar syntax.

#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use crate::def_package;
use crate::plugin::*;
use crate::{Dynamic, Array};
use crate::types::plots::RhaiPlot;

def_package! {
    /// Package of plotting utilities using ruviz.
    pub BasicPlotsPackage(lib) {
        lib.set_standard_lib(true);

        // Register the Plot type
        lib.set_custom_type::<RhaiPlot>("Plot");

        // Register plotting functions
        combine_with_exported_module!(lib, "plots", plots_functions);
    }
}

// Plot operations
#[export_module]
mod plots_functions {
    use crate::types::plots::RhaiPlot;
    use crate::ImmutableString;

    /// Create a new empty plot.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let plot = new_plot();
    /// ```
    #[rhai_fn(return_raw)]
    pub fn new_plot() -> RhaiResult {
        Ok(Dynamic::from(RhaiPlot::new()))
    }

    /// Add a line plot to the current plot.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    /// let y = [0.0, 1.0, 4.0, 9.0, 16.0];
    /// let plot = new_plot();
    /// plot = line(plot, x, y);
    /// ```
    #[rhai_fn(name = "line", return_raw)]
    pub fn line(plot: &mut RhaiPlot, x: Array, y: Array) -> RhaiResult {
        let x_vec: Result<Vec<f64>, String> = x.iter()
            .map(|v| v.as_float().map_err(|typ| format!("X values must be numbers, got {}", typ)))
            .collect();
        let y_vec: Result<Vec<f64>, String> = y.iter()
            .map(|v| v.as_float().map_err(|typ| format!("Y values must be numbers, got {}", typ)))
            .collect();

        let x_vec = x_vec?;
        let y_vec = y_vec?;

        Ok(Dynamic::from(plot.line_internal(x_vec, y_vec)?))
    }

    /// Add a scatter plot to the current plot.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let x = [0.0, 1.0, 2.0, 3.0];
    /// let y = [1.5, 2.3, 1.8, 3.1];
    /// let plot = new_plot();
    /// plot = scatter(plot, x, y);
    /// ```
    #[rhai_fn(name = "scatter", return_raw)]
    pub fn scatter(plot: &mut RhaiPlot, x: Array, y: Array) -> RhaiResult {
        let x_vec: Result<Vec<f64>, String> = x.iter()
            .map(|v| v.as_float().map_err(|typ| format!("X values must be numbers, got {}", typ)))
            .collect();
        let y_vec: Result<Vec<f64>, String> = y.iter()
            .map(|v| v.as_float().map_err(|typ| format!("Y values must be numbers, got {}", typ)))
            .collect();

        let x_vec = x_vec?;
        let y_vec = y_vec?;

        Ok(Dynamic::from(plot.scatter_internal(x_vec, y_vec)?))
    }

    /// Add a bar chart to the current plot.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let labels = ["A", "B", "C"];
    /// let values = [1.0, 2.5, 1.8];
    /// let plot = new_plot();
    /// plot = bar(plot, labels, values);
    /// ```
    #[rhai_fn(name = "bar", return_raw)]
    pub fn bar(plot: &mut RhaiPlot, labels: Array, values: Array) -> RhaiResult {
        let label_vec: Result<Vec<String>, String> = labels.iter()
            .map(|v| {
                v.clone().into_immutable_string()
                    .map(|s: ImmutableString| s.to_string())
                    .map_err(|typ| format!("Labels must be strings, got {}", typ))
            })
            .collect();
        let value_vec: Result<Vec<f64>, String> = values.iter()
            .map(|v| v.as_float().map_err(|typ| format!("Values must be numbers, got {}", typ)))
            .collect();

        let label_vec = label_vec?;
        let value_vec = value_vec?;

        Ok(Dynamic::from(plot.bar_internal(label_vec, value_vec)?))
    }

    /// Set the title of the plot.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let plot = new_plot();
    /// plot = title(plot, "My Plot");
    /// ```
    #[rhai_fn(name = "title", return_raw)]
    pub fn title(plot: &mut RhaiPlot, title: &str) -> RhaiResult {
        Ok(Dynamic::from(plot.title_internal(title)?))
    }

    /// Set the x-axis label of the plot.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let plot = new_plot();
    /// plot = xlabel(plot, "Time (s)");
    /// ```
    #[rhai_fn(name = "xlabel", return_raw)]
    pub fn xlabel(plot: &mut RhaiPlot, label: &str) -> RhaiResult {
        Ok(Dynamic::from(plot.xlabel_internal(label)?))
    }

    /// Set the y-axis label of the plot.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let plot = new_plot();
    /// plot = ylabel(plot, "Amplitude");
    /// ```
    #[rhai_fn(name = "ylabel", return_raw)]
    pub fn ylabel(plot: &mut RhaiPlot, label: &str) -> RhaiResult {
        Ok(Dynamic::from(plot.ylabel_internal(label)?))
    }

    /// Save the plot to a file.
    ///
    /// # Example
    ///
    /// ```rhai
    /// let plot = new_plot();
    /// plot = line(plot, x, y);
    /// plot = title(plot, "My Plot");
    /// save(plot, "output.png");
    /// ```
    #[rhai_fn(name = "save", return_raw)]
    pub fn save(plot: &mut RhaiPlot, path: &str) -> RhaiResult {
        plot.save_internal(path)?;
        Ok(Dynamic::UNIT)
    }
}
