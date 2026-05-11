import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
import math
import datetime
from io import BytesIO

# =============================================================================
# 유틸 함수 (기존 로직 유지)
# =============================================================================

def rotMat3(base_pos, moved_pos, theta):
    rad = np.deg2rad(theta)
    return [
        base_pos[0] + np.cos(rad) * (moved_pos[0] - base_pos[0]) - np.sin(rad) * (moved_pos[1] - base_pos[1]),
        base_pos[1] + np.sin(rad) * (moved_pos[0] - base_pos[0]) + np.cos(rad) * (moved_pos[1] - base_pos[1])
    ]

def line_intersection(a1, b1, a2, b2):
    for i in range(len(a1) - 1):
        for j in range(len(a2) - 1):
            x1, y1 = a1[i], b1[i]
            x2, y2 = a1[i + 1], b1[i + 1]
            x3, y3 = a2[j], b2[j]
            x4, y4 = a2[j + 1], b2[j + 1]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0: continue
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            if (min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4)):
                return (intersect_x, intersect_y)
    return None

def point_to_segment_distance(x1, y1, x2, y2, a, b):
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx**2 + dy**2
    if length_sq == 0: return math.sqrt((a - x1)**2 + (b - y1)**2)
    t = max(0, min(1, ((a - x1) * dx + (b - y1) * dy) / length_sq))
    return math.sqrt((a - (x1 + t * dx))**2 + (b - (y1 + t * dy))**2)

def point_to_arc_distance(px, py, cx, cy, r, theta1, theta2):
    dx, dy = px - cx, py - cy
    dist_to_center = math.sqrt(dx**2 + dy**2)
    angle = math.degrees(math.atan2(dy, dx)) % 360
    if theta1 <= angle <= theta2: return abs(dist_to_center - r)
    s_x, s_y = cx + r * math.cos(math.radians(theta1)), cy + r * math.sin(math.radians(theta1))
    e_x, e_y = cx + r * math.cos(math.radians(theta2)), cy + r * math.sin(math.radians(theta2))
    return min(math.sqrt((px - s_x)**2 + (py - s_y)**2), math.sqrt((px - e_x)**2 + (py - e_y)**2))

def circumcircle(points):
    return max(math.sqrt(x**2 + y**2) for x, y in points)

# =============================================================================
# 핵심 계산 로직
# =============================================================================

def get_params(p_dict):
    c, d, j, k, b, a, f, e, g = p_dict['c'], p_dict['d'], p_dict['j'], p_dict['k'], p_dict['b'], p_dict['a'], p_dict['f'], p_dict['e'], p_dict['g']
    Radius_Stator = c / 100
    Radius_Rotor = d / 100
    Diameter_Aperture_Max = j / 100
    Diameter_Aperture_Min = k / 1000
    Blade_N = b
    Angle_Stator = 270 - 180 / Blade_N
    Angle_Stator_H = 90 - 180 / Blade_N
    Angle_Rotor = a / 10 / 2
    Angle_Blade_Point_1 = f / 10
    Angle_Blade_Point_2 = e / 10
    Angle_Blade_Point_3 = -60
    Size_Max = 13.3 - 0.6 - 0.2 - 0.17
    Size_Min = 5.5 + 0.6 + 0.2 + 0.17
    Radius_Blade_Edge = 0.1

    Pos_Stator_Boss = [Radius_Stator * np.cos(np.deg2rad(Angle_Stator)), Radius_Stator * np.sin(np.deg2rad(Angle_Stator))]
    Pos_Rotor_Boss = [Radius_Rotor * np.cos(np.deg2rad(Angle_Rotor)), Radius_Rotor * np.sin(np.deg2rad(Angle_Rotor))]

    Angle_Line = (np.rad2deg(np.acos(2 * ((Diameter_Aperture_Max - Diameter_Aperture_Min) / 2) / (2 * Radius_Stator) 
                  - np.cos(np.deg2rad(Angle_Stator_H - Angle_Blade_Point_2)))) + Angle_Stator_H - Angle_Blade_Point_2) / 2
    
    Length_Extention_1 = g / 100
    Length_Extention_2 = (Diameter_Aperture_Max - Diameter_Aperture_Min) / 2 * np.tan(np.deg2rad(Angle_Stator_H - Angle_Blade_Point_2 - Angle_Line))
    Angle_Blade_Rotation = 180 - 2 * Angle_Line

    p2 = [Diameter_Aperture_Min / 2 * np.cos(np.deg2rad(Angle_Blade_Point_1)), Diameter_Aperture_Min / 2 * np.sin(np.deg2rad(Angle_Blade_Point_1))]
    p3 = [Diameter_Aperture_Min / 2 * np.cos(np.deg2rad(Angle_Blade_Point_2)), Diameter_Aperture_Min / 2 * np.sin(np.deg2rad(Angle_Blade_Point_2))]
    p1 = [p2[0] - Length_Extention_1 * np.sin(np.deg2rad(Angle_Blade_Point_1)), p2[1] + Length_Extention_1 * np.cos(np.deg2rad(Angle_Blade_Point_1))]
    pe = [(Diameter_Aperture_Min/2 + Radius_Blade_Edge) * np.cos(np.deg2rad(Angle_Blade_Point_1)) - Length_Extention_1 * np.sin(np.deg2rad(Angle_Blade_Point_1)),
          (Diameter_Aperture_Min/2 + Radius_Blade_Edge) * np.sin(np.deg2rad(Angle_Blade_Point_1)) + Length_Extention_1 * np.cos(np.deg2rad(Angle_Blade_Point_1))]
    p4 = [p3[0] + Length_Extention_2 * np.cos(np.deg2rad(270 + Angle_Blade_Point_2)), p3[1] + Length_Extention_2 * np.sin(np.deg2rad(270 + Angle_Blade_Point_2))]
    p5 = [pe[0] + Radius_Blade_Edge * np.cos(np.deg2rad(Angle_Blade_Point_1 + 30)), pe[1] + Radius_Blade_Edge * np.sin(np.deg2rad(Angle_Blade_Point_1 + 30))]
    p6 = [p5[0] + 1.5 * np.cos(np.deg2rad(Angle_Blade_Point_1 - 60)), p5[1] + 1.5 * np.sin(np.deg2rad(Angle_Blade_Point_1 - 60))]

    return locals()

# =============================================================================
# UI 구성
# =============================================================================

st.set_page_config(page_title="Variable_Aperture", layout="wide")
st.title("Variable_Aperture")

with st.sidebar:
    st.header("Settings")
    params = {}
    params['a'] = st.slider("Slot Angle", -50.0, 50.0, -41.0, 0.1) * 10
    params['b'] = st.slider("Blade N", 4, 10, 9)
    params['c'] = st.slider("Radius Stator", 5.1, 10.0, 6.9, 0.01) * 100
    params['d'] = st.slider("Radius Rotor", 4.0, 9.5, 5.8, 0.01) * 100
    params['e'] = st.slider("Blade Angle2", -80.0, 20.0, -30.0, 0.1) * 10
    params['f'] = st.slider("Blade Angle1", -60.0, 30.0, -7.0, 0.1) * 10
    params['g'] = st.slider("Extension Length", 0.0, 3.0, 1.0, 0.01) * 100
    params['h'] = st.slider("Rotation Rotor", 0.0, 10.0, 0.0, 0.1) * 10
    params['i'] = st.slider("Layer", 2, 4, 3)
    params['j'] = st.slider("Diameter Aperture Max", 4.0, 9.0, 8.12, 0.01) * 100
    params['k'] = st.slider("Diameter Aperture Min", 1.0, 3.0, 1.93, 0.001) * 1000
    params['m'] = st.slider("Soma", 5.0, 8.0, 7.82, 0.01) * 100
    reverse_slot = st.checkbox("Reverse Slot", value=False)

p = get_params(params)

tab1, tab2, tab3 = st.tabs(["Design View", "Gaps & Analysis", "Data Export"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Iris Layout")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.add_patch(patches.Circle([0, 0], p["Diameter_Aperture_Min"] / 2, fc='none', ec='k', ls='--', alpha=0.5))
        ax.add_patch(patches.Circle([0, 0], p["Diameter_Aperture_Max"] / 2, fc='none', ec='k', ls='--', alpha=0.5))
        ax.add_patch(patches.Circle([0, 0], p["Size_Max"] / 2, fc='none', ec='y', ls='-', alpha=1))
        
        colors = ['k', 'b', 'g', 'r']
        layer = int(p['i'])
        blade_n = int(p['b'])
        
        for j_idx in range(layer):
            for i_idx in range(int(blade_n / layer)):
                transform2 = Affine2D().rotate_deg_around(0, 0, (360 / blade_n) * (layer * i_idx + 1 + j_idx))
                transform3 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], -p["Angle_Blade_Rotation"] / 10 * params['h'] / 10)
                combined = transform3 + transform2 + ax.transData
                
                ax.add_line(Line2D([p["p1"][0], p["p2"][0]], [p["p1"][1], p["p2"][1]], c=colors[j_idx], transform=combined))
                ax.add_patch(patches.Arc([0, 0], width=p["Diameter_Aperture_Min"], height=p["Diameter_Aperture_Min"],
                                        angle=p["Angle_Blade_Point_2"], theta1=0, theta2=p["Angle_Blade_Point_1"] - p["Angle_Blade_Point_2"],
                                        fc='none', ec=colors[j_idx], linewidth=1.2, transform=combined))
                ax.add_line(Line2D([p["p3"][0], p["p4"][0]], [p["p3"][1], p["p4"][1]], c=colors[j_idx], transform=combined))

        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        st.pyplot(fig)

    with col2:
        st.subheader("Blade Shape")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.add_line(Line2D([p["p1"][0], p["p2"][0]], [p["p1"][1], p["p2"][1]], c='k'))
        ax2.add_patch(patches.Arc([0, 0], width=p["Diameter_Aperture_Min"], height=p["Diameter_Aperture_Min"],
                                  angle=p["Angle_Blade_Point_2"], theta1=0, theta2=p["Angle_Blade_Point_1"] - p["Angle_Blade_Point_2"],
                                  fc='none', ec='k', linewidth=1.2))
        ax2.add_line(Line2D([p["p3"][0], p["p4"][0]], [p["p3"][1], p["p4"][1]], c='k'))
        ax2.set_xlim(-4, 2); ax2.set_ylim(-4, 2); ax2.set_aspect('equal')
        st.pyplot(fig2)

with tab2:
    st.subheader("Gap Analysis")
    # Gap calculation logic simplified for web display
    st.info("실시간 Gap 계산 및 Roundness 곡선 분석 섹션입니다.")
    
    # Placeholder for roundness curve
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    # ... (roundness calculation would go here)
    ax4.set_title("Roundness Curve Placeholder")
    st.pyplot(fig4)

with tab3:
    st.subheader("Export Data")
    if st.button("Generate Excel Report"):
        output = BytesIO()
        df_sample = pd.DataFrame({"Parameter": params.keys(), "Value": params.values()})
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_sample.to_excel(writer, sheet_name='Parameters')
        st.download_button(label="Download Excel", data=output.getvalue(), file_name="iris_design_report.xlsx")

st.markdown("---")
st.caption("Variable_Aperture - Developed for Web using Streamlit")
