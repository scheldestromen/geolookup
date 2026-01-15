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

from geoprofile import Section

PLOT_KWARGS = {
    "coneResistance": {"line_color": "black", "factor": 2},
    "frictionRatioComputed": {"line_color": "red", "factor": 4},
    "porePressureU2": {"line_color": "blue", "factor": 100},
}


def profile_line_from_dps(gdf_hm, dp_list, distance_col='HMTAFSTNL'):
    """
    Create a LineString profile from a list of dike poles (dp) in a GeoDataFrame.
    Returns the LineString and optionally the starting distance value.
    Args:
        gdf_hm: GeoDataFrame with dike pole data
        dp_list: List of dike pole numbers
        distance_col: Column name for distance reference
    Returns:
        line: LineString object
        x0: Starting distance value (if distance_col is provided)
    """
    gdf_hm_subset = gdf_hm[gdf_hm['dp'].isin(dp_list)]
    xy_list = list(zip(gdf_hm_subset['x'], gdf_hm_subset['y']))
    line = LineString(xy_list)
    if distance_col is not None:
        x0 = gdf_hm_subset.iloc[0][distance_col]
        return line, x0
    else:
        return line


def plot_pbs_in_fig(fig, df_metainfo, dp_list):
    """
    Plot peilbuizen (groundwater standpipes) in a Plotly figure for a given dike pole range.
    Args:
        fig: Plotly figure object
        df_metainfo: DataFrame with peilbuis metadata
        dp_list: List of dike pole numbers
    """
    df_metainfo_plot = df_metainfo[df_metainfo.dp.between(
        min(dp_list), max(dp_list))]
    for pos, group in df_metainfo_plot.groupby('position'):
        fig.add_scatter(
            x=group.dp * 100,
            y=group.screen_top,
            mode='markers',
            marker=dict(size=12),
            name=f'peilbuis {pos}<BR>bovenkant filter'
        )


def plot_geoprofile(cols,  gdf_hm, dp_list,  df_metainfo=None, buffer=100, projectname=None, width=1900, height=937, dp_format='.1f', title_suffix='', profile_line=None, plot_path=None):
    """
    Plot a geoprofile along a dike trajectory using soil and groundwater data.
    Args:
        cols: List of geoprofile columns
        gdf_hm: GeoDataFrame with dike pole data
        dp_list: List of dike pole numbers
        df_metainfo: DataFrame with peilbuis metadata (optional)
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
        profile_line, x0 = profile_line_from_dps(gdf_hm, dp_list)
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

    bounds = profile_line.bounds
    gdf_hm_subset = gdf_hm[
        (gdf_hm['x'] >= bounds[0] - buffer) &
        (gdf_hm['x'] <= bounds[2] + buffer) &
        (gdf_hm['y'] >= bounds[1] - buffer) &
        (gdf_hm['y'] <= bounds[3] + buffer) &
        (gdf_hm['dp'].isin(dp_list))
    ]
    gdf_hm_subset.plot(
        ax=axis,
        marker='o',
        color='red',
        markersize=30,
        label='_nolegend_'
    )
    for idx, row in gdf_hm_subset.iterrows():
        axis.text(
            row['x'],
            row['y'],
            f"dp{row['dp']:0.0f}",
            fontsize=8,
            ha='center',
            va='center',
            color='white',
            fontweight='bold',
        )
    plt.savefig(f"{plot_path}{projectname}_map.png")

    # plot section
    fig = profile.plot(
        plot_kwargs=PLOT_KWARGS,
        hue="uniform",
        fillpattern=False,
        surface_level=False,
        groundwater_level=False,
        x0=x0,
    )
    # plot standpipes
    if df_metainfo is not None:
        plot_pbs_in_fig(fig, df_metainfo, dp_list)
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(
            range(int(x0), int(x0) + int(profile_line.length) + 100, 100)),
        ticktext=[f'dp{val/100:{dp_format}}<br>{val}' for val in range(
            int(x0), int(x0) + int(profile_line.length) + 100, 100)]
    )
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


def select_and_plot_deltaversterking(dp_center, fig, df_meta_deltaversterking):
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
    logging.debug(
        f'Closest dp with Deltaversterking to {dp_center} is {closest_dp_dsn}, fn: {fn[-20:]}')

    plot_deltaversterking_background(fig, fn, closest_dp_dsn)

    return closest_dp_dsn


def select_and_plot_surfacelevelprofile(dp_center, fig, df_profiles, delta_dp=1, region='Os'):
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
        (df_profiles.Gebied == region)
    ]
    # find closest dp in df_profiles
    closest_dp_profile = df_profiles_subset.dp.iloc[(
        df_profiles_subset.dp - dp_center).abs().argsort()[:1]].values[0]
    logging.debug(
        f'Closest dp with surface level profile to {dp_center} is {closest_dp_profile}')

    # prepare plotting
    lst_plot_profiel = ['MV.bin', 'Sloot.1a', 'Sloot.1c', 'Sloot.1d', 'Sloot.1b', 'Weg.1', 'Teen.1', 'Berm.1a',
                        'Berm.1b', 'Kruin.1', 'Kruin.2', 'Berm.2a', 'Berm.2b', 'Teen.2', 'Weg.2', 'Sloot.2', 'MVB.bui']
    lst_plot_profiel_x = ['x' + s for s in lst_plot_profiel]
    lst_plot_profiel_y = ['y' + s for s in lst_plot_profiel]
    lst_plot_profiel_y[-1] = 'yMV.bui'
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
            )
        else:
            logging.debug(
                f"Not enough data to plot profile for dp{row['dp']:.1f}")

    return closest_dp_profile


def create_figure_for_dp(dp_center, df_meta_deltaversterking, geoprofile_cols, df_profiles, df_gwl, delta_dp=1, plot_path=None, xmin=30, xmax=30):
    """
    Create a Plotly figure for a specific dike pole, including soil, surface, and groundwater profiles.
    Args:
        dp_center: Central dike pole number
        df_meta_deltaversterking: DataFrame with Deltaversterking metadata
        geoprofile_cols: List of geoprofile columns
        df_profiles: DataFrame with surface level profiles
        df_gwl: DataFrame with groundwater levels
        delta_dp: Range around central dike pole
        plot_path: Path for saving plot
        xmin: Minimum x-axis value
        xmax: Maximum x-axis value
    Returns:
        fig: Plotly figure object
        fn: Filename of saved plot
    """

    # create figure
    fig = make_subplots(
        rows=1,
        cols=1,
        y_title="(m NAP)",
        shared_yaxes=True,
        horizontal_spacing=0.01,
    )

    closest_dp_dsn = select_and_plot_deltaversterking(
        dp_center, fig, df_meta_deltaversterking)

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
        dp_center, fig, df_profiles, delta_dp=delta_dp, region='Os')

    # PLOT GROUNDWATER STANDPIPES
    plot_pbs_in_fig(
        fig,
        df_gwl.loc[df_gwl.dp.between(
            dp_center - delta_dp, dp_center + delta_dp) & (df_gwl.region == 'Os')],
        dp_center
    )

    # LAYOUT ADJUSTMENTS
    fig.update_layout(
        title_text=f"profiel rondom dp{dp_center} ± {delta_dp*100:0.0f}m, tekening Deltaversterking bij dp{closest_dp_dsn} ({(closest_dp_dsn-dp_center)*100:+0.0f}m)")
    fig.update_xaxes(
        range=[-xmin, xmax], title_text="afstand tot referentie lijn [m]")
    # fig.update_xaxes(autorange="reversed")
    fn = fr"{plot_path}\profiel_dp{dp_center}__plusmin{delta_dp}.html"
    pio.write_html(fig, file=fn, auto_open=True)

    return fig, fn


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


def plot_deltaversterking_background(figure, fn, dp=None):
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

    figure.add_layout_image(
        dict(
            source=img_base64,
            xref="x",
            yref="y",
            x=x_min,
            y=y_max,
            sizex=x_max - x_min,
            sizey=y_max - y_min,
            sizing="stretch",
            opacity=0.5,
            layer="below",
        )
    )


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


def plot_pbs_in_fig(fig, df_metainfo, dp_center):
    """
    Plot peilbuizen (groundwater standpipes) in a Plotly figure for a specific dike pole.
    Args:
        fig: Plotly figure object
        df_metainfo: DataFrame with peilbuis metadata
        dp_center: Central dike pole number
    """
    for pos, group in df_metainfo.groupby('position'):
        for _, row in group.iterrows():
            color = dp_to_color(row.dp, dp_center, color_center='darkgreen')
            fig.add_trace(
                go.Scatter(
                    x=[row.distance_to_ref, row.distance_to_ref],
                    y=[row.screen_top, row.screen_bottom],
                    mode="lines",
                    line=dict(color=color, width=10),
                    name=f'filter {pos}<BR>dp{row.dp}, {row.timeseries_id}',
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[row.distance_to_ref, row.distance_to_ref],
                    y=[row.screen_top, row.tube_top],
                    mode="lines",
                    line=dict(color=color, width=1),
                    showlegend=False,
                )
            )
        fig.add_scatter(
            x=group.distance_to_ref,
            y=group.tube_top,
            mode='markers',
            marker=dict(
                color=dp_to_color(row.dp, dp_center, color_center='darkgreen'),
                size=12,
                symbol='diamond'
            ),
            name=f'peilbuis {pos}<BR>dp{row.dp}, bovenkant buis'
        )
