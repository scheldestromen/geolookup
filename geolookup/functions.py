import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from shapely.geometry import LineString
from affine import Affine

import plotly
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go

import rasterio
import base64
from io import BytesIO
import PIL.Image
from datetime import datetime

from geoprofile import Section

PLOT_KWARGS = {
    "coneResistance": {"line_color": "black", "factor": 2},
    "frictionRatioComputed": {"line_color": "red", "factor": 4},
    "porePressureU2": {"line_color": "blue", "factor": 100},
}


def describe_range(min, max, return_string=True):
    """
    Describe the range between two values, returning either a formatted string or tuple.
    The center is rounded to the nearest 10. The span is given in meters if less than 1000, otherwise in kilometers.
    Args:
        min (float): Minimum value.
        max (float): Maximum value.
        return_string (bool): If True, returns a formatted string; otherwise returns a tuple (center, span, span_unit).
    Returns:
        str or tuple: Description of the range as a string or (center, span, span_unit).
    """
    # round center to nearest 10
    center = round(((min + max) / 2) / 10) * 10
    span = max - min
    if span < 1000:
        span_unit = 'm'
        # round span to nearest 10
        span = round(span / 10) * 10
    else:
        span_unit = 'km'
        span = round(span / 1000, 0)
    if return_string:
        return f'centered at {center:.0f} with span {span:0.0f} {span_unit}'
    else:
        return center, span, span_unit


def describe_extent(cols):
    """
    Calculate the extent (min and max) in x and y from a list of columns with x and y attributes.
    Logs the extent using describe_range for both axes.
    Args:
        cols (list): List of objects with 'x' and 'y' attributes.
    Returns:
        tuple: (x_min, x_max, y_min, y_max)
    """
    x_min = min([item.x for item in cols])
    x_max = max([item.x for item in cols])
    y_min = min([item.y for item in cols])
    y_max = max([item.y for item in cols])

    logging.info(
        f'Data X: {describe_range(x_min, x_max)}, Y: {describe_range(y_min, y_max)}')

    return x_min, x_max, y_min, y_max


def describe_oc(
        oc,
        expected_obs_interval=10,  # interval in minutes
):
    expected_obs_per_day = (24 * 60) / expected_obs_interval
    df_period_obs_approx_years = (oc.stats.get_no_of_observations(
    ) / (expected_obs_per_day))/365

    logging.info(f'Obs Collection has {len(oc)} monitoring wells, on {oc.location.unique().size} locations. First observation {oc.stats.dates_first_obs.min().strftime("%Y-%m-%d")}, last observation {oc.stats.dates_last_obs.max().strftime("%Y-%m-%d")}')
    logging.info(
        f'Obs Collection has on average {df_period_obs_approx_years.mean():.1f} years of observations per monitoring well, median {df_period_obs_approx_years.median():.1f} years. Assuming observations interval each {expected_obs_interval} minutes.')

    logging.info(
        f'Data X: {describe_range(oc.x.min(), oc.x.max())}, Y: {describe_range(oc.y.min(), oc.y.max())}')

    if not oc.index.is_unique:
        logging.warning("Non-unique index values:")
        logging.warning(oc.index[oc.index.duplicated()])
    else:
        logging.info("Index is unique.")


def describe_columns(cols):
    """
    Log the number of columns and their spatial extent.
    Args:
        cols (list): List of columns/objects with 'x' and 'y' attributes.
    """
    logging.info(f'Number of columns: {len(cols)}')
    # loop over all cols, for each item check if 'dp' attribute exists, if so collect dp values
    dps = [col.dp for col in cols if hasattr(col, 'dp')]
    if len(dps) > 0:
        unique_dps = set(dps)
        logging.info(
            f'Dike pole range: dp{min(unique_dps):.1f} - dp{max(unique_dps):.1f}, total {len(unique_dps)} unique dike poles.')
    describe_extent(cols)


def profile_line_from_dps(df_hm, dp_list, distance_col='dp', distance_factor=100):
    """
    Create a LineString profile from a list of dike poles (dp) in a GeoDataFrame.
    Returns the LineString and optionally the starting distance value.
    Args:
        df_hm: GeoDataFrame with dike pole data
        dp_list: List of dike pole numbers
        distance_col: Column name for distance reference
    Returns:
        line: LineString object
        x0: Starting distance value (if distance_col is provided)
    """
    df_hm_subset = df_hm[df_hm['dp'].isin(dp_list)]
    xy_list = list(zip(df_hm_subset['x'], df_hm_subset['y']))
    line = LineString(xy_list)
    if distance_col is not None:
        x0 = df_hm_subset.iloc[0][distance_col] * distance_factor
        return line, x0
    else:
        return line


def add_points_to_map(axis, df, profile_line, buffer, dp_list, plot_col='dp', annotate=False, marker='o', color='red', markersize=30, label='_nolegend'):
    bounds = profile_line.bounds

    df_subset = df[
        (df['x'] >= bounds[0] - buffer) &
        (df['x'] <= bounds[2] + buffer) &
        (df['y'] >= bounds[1] - buffer) &
        (df['y'] <= bounds[3] + buffer) &
        (df['dp'].isin(dp_list))
    ]
    df_subset = df_subset.dropna(subset=plot_col)
    logging.info(
        f'Added {len(df_subset)} points to map for dps {dp_list}, type is {type(df_subset)}.')
    if len(df_subset) == 0:
        return
    df_subset[plot_col].plot(
        ax=axis,
        marker=marker,
        color=color,
        markersize=markersize,
        label=label
    )

    if annotate:
        for idx, row in df_subset.iterrows():
            if plot_col == 'dp':
                label = f"dp{row['dp']:0.0f}"
            else:
                label = f"{row[plot_col]}"
            axis.text(
                row['x'],
                row['y'],
                label,
                fontsize=8,
                ha='center',
                va='center',
                color='white',
                fontweight='bold',
            )


def plot_geoprofile(cols,  df_hm, dp_list,  oc_gwl=None, buffer=100, projectname=None, width=1900, height=937, dp_format='.1f', title_suffix='', profile_line=None, plot_path=None, ylims=None, groundwater_level=False, surface_level=False, region='Os'):
    """
    Plot a geoprofile along a dike trajectory using soil and groundwater data.
    Args:
        cols: List of geoprofile columns
        df_hm: GeoDataFrame with dike pole data
        dp_list: List of dike pole numbers
        oc: Observation collection of groundwater standpipes and timeseries (optional)
        buffer: Buffer distance for map plotting
        projectname: Name for output files
        width: Width of output image
        height: Height of output image
        dp_format: Format for dike pole labels
        title_suffix: Suffix for plot title
        profile_line: Optional LineString for profile
        plot_path: Path for saving plots
    Returns:
        fig: Plotly figure object
        profile: Section object
    """
    # fix profile
    if profile_line is None:
        profile_line, x0 = profile_line_from_dps(df_hm, dp_list)
    else:
        x0 = 0
    profile = Section(
        cols,
        sorting_algorithm="nearest_neighbor",
        profile_line=profile_line,
        buffer=buffer,
        reproject=True,
    )

    # plot map
    axis = profile.plot_map(
        add_basemap=True, add_tags=False, tag_type="index", show_all=True
    )
    axis.set_xlim(profile_line.bounds[0] -
                  buffer, profile_line.bounds[2] + buffer)
    axis.set_ylim(profile_line.bounds[1] -
                  buffer, profile_line.bounds[3] + buffer)
    add_points_to_map(axis, df_hm, profile_line, buffer,
                      dp_list, plot_col='dp', annotate=True, marker='o', color='red', label='dijkpaal')
    if oc_gwl is not None:
        oc_gwl_plot = oc_gwl.loc[oc_gwl.dp.between(
            min(dp_list), max(dp_list)) & (oc_gwl.region == region)
        ]
        add_points_to_map(axis, oc_gwl_plot, profile_line, buffer,
                          dp_list, plot_col='dp', annotate=True, marker='s', color='blue', markersize=20, label='peilbuis')

    axis.legend(loc='best')
    plt.savefig(f"{plot_path}{projectname}_map.png")

    # plot section
    fig = profile.plot(
        plot_kwargs=PLOT_KWARGS,
        hue="uniform",
        fillpattern=False,
        surface_level=surface_level,
        groundwater_level=groundwater_level,
        x0=x0,
    )
    # plot standpipes
    if oc_gwl_plot is not None:
        plot_standpipes_in_fig(
            fig,
            oc_gwl_plot,
            method='geoprofile',
        )

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(
            range(int(x0), int(x0) + int(profile_line.length) + 100, 100)),
        ticktext=[f'dp{val/100:{dp_format}}<br>{val}' for val in range(
            int(x0), int(x0) + int(profile_line.length) + 100, 100)]
    )
    if ylims is not None:
        fig.update_yaxes(range=ylims)

    fig.update_layout(title_text=projectname + title_suffix)
    _ = plotly.offline.plot(
        fig, filename=f"{plot_path}{projectname}_webpage.html")

    fig.write_image(f"{plot_path}{projectname}_profile.png",
                    width=width, height=height)

    return fig, profile


def update_offset(fn_img, transform_matrix, replace_search='_flipped', replace_with='_flipped_offset'):
    """
    Update the affine transform matrix with an offset from a file, if available.
    Args:
        fn_img: Filename of image
        transform_matrix: Affine transform matrix
        replace_search: String to search in filename
        replace_with: String to replace in filename
    Returns:
        transform_matrix: Updated affine transform matrix
    """
    if os.path.exists(fn_img.replace(replace_search, replace_with)):
        df_offset = pd.read_csv(
            fn_img.replace(replace_search, replace_with),
            sep=' ',
            names=['parameter', 'value',],
            header=None,
            comment='#',
            index_col=0
        )
        if 'x' in df_offset.index:
            offset_x = df_offset.loc['x', 'value']
            transform_matrix = transform_matrix * \
                Affine.translation(offset_x, 0)
            print(offset_x)
        else:
            logging.warning(
                f'offset file found, but no x parameter. Not expected at {fn_img.replace(replace_search, replace_with)}. No offset applied.')
    else:
        logging.debug(
            f'No offset file found at {fn_img.replace(replace_search, replace_with)}, no offset applied.')
    return transform_matrix


def select_and_plot_deltaversterking(dp_center, fig, df_meta_deltaversterking, fig_row=1, fig_col=1):
    """
    Select the closest dike pole with Deltaversterking and plot its background in the figure.
    Args:
        dp_center: Central dike pole number
        fig: Plotly figure object
        df_meta_deltaversterking: DataFrame with Deltaversterking metadata
    Returns:
        closest_dp_dsn: Closest dike pole with Deltaversterking
    """

    # find closest dp in df_meta_deltaversterking
    df_met_dsn = df_meta_deltaversterking.dropna(subset=['dsn_file_tif'])
    closest_dp_dsn = df_met_dsn.dp.iloc[(
        df_met_dsn.dp - dp_center).abs().argsort()[:1]].values[0]
    fn = df_met_dsn.loc[df_met_dsn.dp ==
                        closest_dp_dsn, 'dsn_file_tif'].values[0]
    if 'x_offset' in df_meta_deltaversterking.columns:
        x_offset = df_met_dsn.loc[df_met_dsn.dp ==
                                  closest_dp_dsn, 'x_offset'].values[0]
        if pd.notnull(x_offset):
            pass
        else:
            x_offset = 0

    logging.debug(
        f'Closest dp with Deltaversterking to {dp_center} is {closest_dp_dsn}, fn: {fn[-20:]}, x_offset: {x_offset}')

    plot_deltaversterking_background(
        fig, fn, closest_dp_dsn, fig_row=fig_row, fig_col=fig_col, x_offset=x_offset)

    return closest_dp_dsn


def select_and_plot_surfacelevelprofile(dp_center, fig, df_profiles, delta_dp=1, region='Os', fig_row=1, fig_col=1):
    """
    Select and plot the surface level profiles  in the specified region.
    Args:
        dp_center: Central dike pole number
        fig: Plotly figure object
        df_profiles: DataFrame with surface level profiles
        delta_dp: Range around central dike pole
        region: Region name
    Returns:
        closest_dp_profile: Closest dike pole with surface level profile
    """

    # select data
    df_profiles_subset = df_profiles[
        df_profiles.dp.between(dp_center - delta_dp, dp_center + delta_dp) &
        (df_profiles.region == region)
    ]
    # find closest dp in df_profiles
    closest_dp_profile = df_profiles_subset.dp.iloc[(
        df_profiles_subset.dp - dp_center).abs().argsort()[:1]].values[0]
    logging.debug(
        f'Closest dp with surface level profile to {dp_center} is {closest_dp_profile}')

    # prepare plotting
    lst_plot_profiel = ['mv.bin', 'sloot.1a', 'sloot.1c', 'sloot.1d', 'sloot.1b', 'weg.1', 'teen.1', 'berm.1a',
                        'berm.1b', 'kruin.1', 'kruin.2', 'berm.2a', 'berm.2b', 'teen.2', 'weg.2', 'sloot.2', 'mvb.bui']
    lst_plot_profiel_x = ['x' + s for s in lst_plot_profiel]
    lst_plot_profiel_y = ['y' + s for s in lst_plot_profiel]
    lst_plot_profiel_y[-1] = 'ymv.bui'
    lst_plot_profiel_y[3] = 'ysloot.1d'

    for index, row in df_profiles_subset.iterrows():
        # Plot the profile line in the figure
        x_vals = [row[x]
                  for x in lst_plot_profiel_x if x in row and pd.notnull(row[x])]
        y_vals = [row[y]
                  for y in lst_plot_profiel_y if y in row and pd.notnull(row[y])]
        if len(x_vals) > 1 and len(y_vals) > 1:
            line_color = dp_to_color(row['dp'], closest_dp_profile)
            fig.add_scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=f"profiel dp{row['dp']:.1f}",
                line=dict(
                    width=4 if row['dp'] == closest_dp_profile else 2, color=line_color),
                row=fig_row,
                col=fig_col,
            )
        else:
            logging.debug(
                f"Not enough data to plot profile for dp{row['dp']:.1f}")

    return closest_dp_profile


def create_figure_for_dp(dp_center, df_meta_deltaversterking, geoprofile_cols, df_profiles, oc_gwl, df_gmw=None, delta_dp=1, width=1900, height=1200, region='Os', xmin=-30, xmax=20, plot_path=r'plot\\', plot_gwl=False, ylims_upper=None, auto_open=True):

    # prepare groundwater data
    oc_gwl_plot = oc_gwl.loc[oc_gwl.dp.between(
        dp_center - delta_dp, dp_center + delta_dp
    ) & (oc_gwl.region == region)].copy()

    if plot_gwl:
        fig_rows = 2
        colors = ['red', 'blue', 'limegreen', 'orange',
                  'purple', 'brown', 'pink', 'gray']
        for i, (index, row) in enumerate(oc_gwl_plot.iterrows()):
            oc_gwl_plot.at[index, 'plot_color'] = colors[i % len(colors)]
    else:
        fig_rows = 1

    # create figure
    fig = make_subplots(
        rows=fig_rows,
        cols=1,
        horizontal_spacing=0.01,
    )

    plot_section_for_dp(fig, dp_center=dp_center, df_meta_deltaversterking=df_meta_deltaversterking, geoprofile_cols=geoprofile_cols,
                        df_profiles=df_profiles, oc_gwl=oc_gwl_plot, delta_dp=delta_dp, region=region, xmin=xmin, xmax=xmax, fig_row=1, fig_col=1)

    if df_gmw is not None:
        df_plot = df_gmw.loc[df_gmw.dp.between(
            dp_center - delta_dp, dp_center + delta_dp) & (df_gmw.region == region)]
        logging.debug(f'gmw worden geplot, aantal: {len(df_plot)}')
        plot_standpipes_in_fig(
            fig,
            df_plot,
            [dp_center - delta_dp, dp_center, dp_center + delta_dp],
            dp_center,
            dp_alternative_color='fuchsia',
            method='by_dp_marker',
        )
    else:
        logging.debug('gmw worden niet geplot')

    if plot_gwl:
        # plot observations in separate subplot
        for index, row in oc_gwl_plot.iterrows():
            # mean in upper plot
            fig.add_trace(
                go.Scatter(
                    # x=[row.dp - 0.3, row.dp + 0.3],
                    x=[row.distance_to_ref - 0.8, row.distance_to_ref + 0.8],
                    y=[row.obs.gwl_mnap.mean()]*2,
                    mode="lines",
                    line=dict(color=row.plot_color, width=4) if hasattr(
                        row, 'plot_color') else dict(color='black', width=4),
                    # name=f'pb {row.position}<BR>dp{row.dp}, {row.timeseries_id}<BR>gem:{row.obs.gwl_mnap.mean():.2f} over {row.obs.index.year.min()}-{row.obs.index.year.max()}',
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            # individual observations in lower plot
            fig.add_trace(
                go.Scatter(
                    x=row.obs.index,
                    y=row.obs.gwl_mnap,
                    mode="lines",
                    line=dict(color=row['plot_color'], width=2),
                    # {row.timeseries_id}
                    name=f'pb {row.position} dp{row.dp} {row.filter_letter}',
                ),
                row=2,
                col=1,
            )
            fig.update_xaxes(
                range=[row.obs.index.min(), row.obs.index.max()],
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="grondwaterstand (m NAP)", row=2, col=1)

    fig.update_yaxes(title_text="(m NAP)", row=1, col=1)

    if ylims_upper is not None:
        fig.update_yaxes(range=ylims_upper, row=1, col=1)

    # set lengend below lowest subplot
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Add current date and time in upper right corner
    fig.add_annotation(
        text=datetime.now().strftime("%d-%m-%Y %H:%M"),
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.99,
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="right",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
    )

    fn = fr"{plot_path}\profiel_dp{int(dp_center):04d}__plusmin{delta_dp}.html"
    pio.write_html(fig, file=fn, auto_open=auto_open)
    fig.write_image(fn.replace('.html', '.png'),
                    width=width, height=height)
    return fig, fn


# def create_figure_for_dp(dp_center, df_meta_deltaversterking, geoprofile_cols, df_profiles, oc_gwl, delta_dp=1, width=1900, height=937, region='Os', xmin=-30, xmax=30, plot_path=r'plot\\'):
def plot_section_for_dp(fig, dp_center, df_meta_deltaversterking, geoprofile_cols, df_profiles, oc_gwl, delta_dp=1, region='Os', xmin=-30, xmax=30, fig_row=1, fig_col=1):
    """
    Create a Plotly figure for a specific dike pole, including soil, surface, and groundwater profiles.
    Args:
        dp_center: Central dike pole number
        df_meta_deltaversterking: DataFrame with Deltaversterking metadata
        geoprofile_cols: List of geoprofile columns
        df_profiles: DataFrame with surface level profiles
        oc_gwl: Object containing groundwater levels
        delta_dp: Range around central dike pole
        plot_path: Path for saving plot
        xmin: Minimum x-axis value
        xmax: Maximum x-axis value
    Returns:
        fig: Plotly figure object
        fn: Filename of saved plot
    """

    closest_dp_dsn = select_and_plot_deltaversterking(
        dp_center, fig, df_meta_deltaversterking, fig_row=fig_row, fig_col=fig_col)

    # PLOT SOIL DATA

    # select items from geoprofile_cols, sort on distance to dp_center
    geoprofile_cols_subset = sorted([
        col for col in geoprofile_cols
        if dp_center - delta_dp <= col.dp <= dp_center + delta_dp
    ],
        key=lambda col: abs(col.dp - dp_center)
    )
    # re order, place dp_center first, then others by distance
    # now dp_center is plotted at actual position,
    # others shifted left/right when duplicate distance_to_ref are present
    geoprofile_cols_subset = (
        [col for col in geoprofile_cols_subset if col.dp == dp_center] +
        [col for col in geoprofile_cols_subset if col.dp != dp_center]
    )
    plot_colums_in_figure(geoprofile_cols_subset, fig,
                          dp_highlight=[dp_center])

    # PLOT SURFACE LEVEL PROFILE
    closest_dp_profile = select_and_plot_surfacelevelprofile(
        dp_center, fig, df_profiles, delta_dp=delta_dp, region=region, fig_row=fig_row, fig_col=fig_col)

    # PLOT GROUNDWATER STANDPIPES
    plot_standpipes_in_fig(
        fig,
        oc_gwl.loc[oc_gwl.dp.between(
            dp_center - delta_dp, dp_center + delta_dp) & (oc_gwl.region == region)],
        [dp_center - delta_dp, dp_center, dp_center + delta_dp],
        dp_center,
        color_center='limegreen',
        method='by_dp',
    )

    # LAYOUT ADJUSTMENTS
    d_dp = (closest_dp_dsn-dp_center) * 100
    fig.update_layout(
        title_text=f"profiel rondom dp{dp_center} ± {delta_dp*100:0.0f}m, tekening Deltaversterking bij dp{closest_dp_dsn} ({d_dp:+0.0f}m)",
        title_font=dict(
            color="black" if abs(d_dp) < 150 else "red", size=16),
    )
    fig.update_xaxes(
        range=[xmin, xmax],
        title_text="afstand tot referentie lijn [m]",
        row=fig_row,
        col=fig_col
    )
# fig.update_xaxes(autorange="reversed")


def plot_colums_in_figure(geoprofile_cols_subset, figure, dp_highlight=None):
    """
    Plot geoprofile columns in a Plotly figure, highlighting specified dike poles.
    Args:
        geoprofile_cols_subset: List of geoprofile columns to plot
        figure: Plotly figure object
        dp_highlight: Dike pole(s) to highlight
    """
    if figure is None:
        figure = make_subplots(
            rows=1,
            cols=1,
            y_title="Depth [m REF]",
            shared_yaxes=True,
            horizontal_spacing=0.01,
        )
    plotted_x = []
    for col in geoprofile_cols_subset:
        # Check if plot_x is already taken; if so, find next available integer
        plot_x = col.distance_to_ref.round(0)
        # for lower dps, shift left
        if col.dp < dp_highlight:
            sign = -1
        else:
            sign = 1
        while plot_x in plotted_x:
            plot_x += (1 * sign)
        plotted_x.append(plot_x)
        logging.debug(
            f"Plotting column dp {col.dp}, distance={col.distance_to_ref}, plotted={plotted_x}")

        col.plot(
            figure=figure,
            x0=plot_x,
            d_left=0.1,
            d_right=0.1,
            profile=True,
        )
        # plot marker at z
        show_name = plotted_x == [plot_x]
        figure.add_trace(
            go.Scatter(
                x=[plot_x],
                y=[col.z],
                mode="markers",
                marker=dict(color="darkgray", size=10),
                showlegend=show_name,
                name='maaiveld sondering' if show_name else None,  # add label for legend only once
            ),
            row=1,
            col=1,
        )
        figure.add_annotation(
            x=plot_x,
            y=col.z,
            text=f"dp{col.dp:.1f} ({plot_x - col.distance_to_ref:0.0f})",
            showarrow=False,
            yshift=0,  # No vertical shift, annotation is exactly at (x, y)
            textangle=-90,
            font=dict(
                color="white", size=12),
            bgcolor=dp_to_color(col.dp, dp_highlight),
            bordercolor="black",
            borderwidth=2,
            xanchor="center",
            yanchor="bottom"  # Align annotation above the marker
        )


def plot_deltaversterking_background(figure, fn, dp=None, fig_row=1, fig_col=1, x_offset=0):
    """
    Add Deltaversterking raster background to a Plotly figure.
    Args:
        figure: Plotly figure object
        fn: Filename of raster image
        dp: Dike pole number (optional)
    """
    with rasterio.open(fn) as src:
        bounds = src.bounds
        transform_matrix = src.transform
        image = src.read(1)  # Read the first band if needed

    # Convert raster image to PIL Image for plotly
    img_array = np.array(image)
    img_min, img_max = np.nanmin(img_array), np.nanmax(img_array)
    img_norm = ((img_array - img_min) /
                (img_max - img_min) * 255).astype(np.uint8)
    pil_img = PIL.Image.fromarray(img_norm)

    # Calculate image extent in plot coordinates using bounds and transform
    x_min, x_max = bounds.left, bounds.right
    y_min, y_max = bounds.bottom, bounds.top

    # update_offset(fn_img, transform_matrix)

    # Convert PIL image to PNG bytes and then to base64 string for plotly
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = "data:image/png;base64," + \
        base64.b64encode(img_bytes).decode()

    # print(fig_row, fig_col, f"x{fig_row}", f"y{fig_row}")
    # xref="x1",
    # yref="y1",
    logging.debug(
        f'Adding Deltaversterking background from {fn[-30:]} to figure at row {fig_row}, col {fig_col}.')

    figure.add_layout_image(
        dict(
            source=img_base64,
            # xref=f"x{fig_row}",  # TODO: assumes one column, multiple rows
            # yref=f"y{fig_row}",
            xref="x",
            yref="y",
            x=x_min + x_offset,
            y=y_max,
            sizex=x_max - x_min,
            sizey=y_max - y_min,
            sizing="stretch",
            opacity=0.8,
            layer="below",
        )
    )

    # set max of ylim to y_max
    figure.update_yaxes(range=[None, y_max], row=fig_row, col=fig_col)


def dp_to_color(dp, dp_center, colors_below=["red", "darkred", "tomato", "firebrick"], color_center="green", colors_above=["blue", "royalblue", "deepskyblue", "navy"]):
    """
    Assign a color to a dike pole based on its position relative to the center.
    Args:
        dp: Dike pole number
        dp_center: Central dike pole number
        colors_below: Colors for dike poles below center
        color_center: Color for center dike pole
        colors_above: Colors for dike poles above center
    Returns:
        color: Assigned color string
    """
    if dp == dp_center:
        return color_center
    else:
        if dp < dp_center:
            colors = colors_below
        elif dp > dp_center:
            colors = colors_above
        else:
            # Fallback color
            colors = ["gray"]
        color_index = min(int(abs(dp - dp_center)) - 1, len(colors) - 1)
        return colors[color_index]


def plot_standpipes_in_fig(fig, oc, dp_list=None, dp_center=None, color_center='green', dp_alternative_color='limegreen', method='geoprofile', fig_row=1, fig_col=1):
    """
    Plot groundwater standpipes in a Plotly figure for a given dike pole range.
    Args:
        fig: Plotly figure object
        oc: Observation collection of groundwater standpipes and timeseries
        dp_list: List of dike pole numbers
    """
    if len(oc) == 0:
        logging.warning(
            'No standpipes to plot in figure. ObsCollection is empty.')
        return
    else:
        logging.info(
            f'Plotting {len(oc)} standpipes in figure, method={method}.')

    if method == 'by_dp':
        logging.debug(f'length oc: {len(oc)}')
        logging.debug(f'oc position values: {oc.position.unique()}')
        for pos, group in oc.groupby('position'):
            for index, row in group.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row.distance_to_ref, row.distance_to_ref],
                        y=[row.screen_top, row.screen_bottom],
                        mode="lines",
                        line=dict(color=row.plot_color, width=10) if hasattr(
                            row, 'plot_color') else dict(color=dp_alternative_color, width=10),
                        name=row.label if hasattr(
                            # {row.timeseries_id}'
                            row, 'label') else f'pb {row.position} dp{row.dp} {row.filter_letter}',
                    ),
                    row=fig_row,
                    col=fig_col,
                )
    if method == 'by_dp_marker':
        oc['screen_mid'] = oc[['screen_top', 'screen_bottom']].mean(axis=1)
        counter = 0
        for index, row in oc.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row.distance_to_ref],
                    y=[row.screen_mid],
                    mode="markers",
                    marker=dict(
                        size=15,
                        symbol=["diamond", "cross", "x", "pentagon",
                                "hexagram", "star-square", "circle-cross"][counter % 7]+"-open",
                        line=dict(color=dp_alternative_color, width=2)
                    ),
                    name=row.label if hasattr(
                        row, 'label') else f'pb {row.position} dp{row.dp} {row.filter_letter}',
                ),
                row=fig_row,
                col=fig_col,
            )
            counter += 1

    elif method == 'geoprofile':
        for index, row in oc.iterrows():
            # color is purple when posititon == kruin; color limegreen when position is binnenteen or binnenberm
            if row.position == 'kruin':
                color = 'purple'
                offset = 0
            elif row.position in ['binnenteen', 'binnenberm']:
                color = 'limegreen'
                offset = 10
            elif row.position == 'buitenberm':
                color = 'orange'
                offset = -10
            else:
                color = 'gray'
                offset = 15
            fig.add_trace(
                go.Scatter(
                    # x=[row.distance_to_ref, row.distance_to_ref],
                    x=[row.dp * 100 + offset]*2,
                    y=[row.screen_top, row.screen_bottom],
                    mode="lines",
                    line=dict(color=row.plot_color, width=10) if hasattr(
                        row, 'plot_color') else dict(color=color, width=10),
                    name=f'filter {row.position}<BR>dp{row.dp}, {row.timeseries_id}',

                ),
                row=fig_row,
                col=fig_col,
            )
            fig.add_trace(
                go.Scatter(
                    # x=[row.distance_to_ref, row.distance_to_ref],
                    x=[row.dp * 100 + offset]*2,
                    y=[row.screen_top, row.tube_top],
                    mode="lines",
                    line=dict(color=row.plot_color, width=1) if hasattr(
                        row, 'plot_color') else dict(color=color, width=1),
                    showlegend=False,

                ),
                row=fig_row,
                col=fig_col,
            )
            print(row.dp * 100 + offset, row.tube_top)
            fig.add_scatter(
                go.Scatter(
                    x=[row.dp * 100 + offset],
                    y=[row.tube_top],
                    mode='markers',
                    marker=dict(
                        color=row.plot_color if hasattr(
                            row, 'plot_color') else color,
                        size=12,
                        symbol='diamond'
                    ),
                    name=f'peilbuis {row.position}<BR>dp{row.dp}, bovenkant buis',
                ),
                row=fig_row,
                col=fig_col,
            )
    else:
        logging.warning(
            f'Unknown method {method} for plotting stand pipes. Nothing plotted.')
