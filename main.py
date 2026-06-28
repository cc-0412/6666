import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
from folium import plugins
import random
import time
import math
import json
import io
from datetime import datetime, timedelta
import copy
import heapq
# 修复语法错误：补充as关键字
import numpy as np
# 新增MAVLink通信依赖
from pymavlink import mavutil

# ==================== 页面基础配置 ====================
st.set_page_config(page_title="无人机地面站系统 - 平行偏移绕行", layout="wide")

# ==================== GCJ02 坐标常量 ====================
SCHOOL_CENTER_GCJ = [118.7490, 32.2340]
DEFAULT_A_GCJ = [118.746956, 32.232945]
DEFAULT_B_GCJ = [118.751589, 32.235204]
DEG_TO_M = 111000.0

# 地图瓦片地址
ARCGIS_SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
AMAP_VECTOR_URL = "https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"

# ==================== 【新增】MAVLink 全局初始化函数 ====================
def init_mavlink_state():
    if "mav_conn" not in st.session_state:
        st.session_state.mav_conn = None
    if "mav_msg_cache" not in st.session_state:
        st.session_state.mav_msg_cache = {}
    if "mav_msg_type" not in st.session_state:
        st.session_state.mav_msg_type = "HEARTBEAT"

# 建立MAVLink UDP连接（适配PX4 SITL 14550端口）
def connect_mavlink(udp_ip="127.0.0.1", port=14550):
    try:
        master = mavutil.mavlink_connection(f"udp:{udp_ip}:{port}")
        master.wait_heartbeat(blocking=True, timeout=3)
        st.session_state.mav_conn = master
        add_gcs_obc_fcu_log(f"MAVLink连接成功 | 设备ID:{master.target_system}")
        return True
    except Exception as e:
        add_gcs_obc_fcu_log(f"MAVLink连接失败：{str(e)}")
        st.session_state.mav_conn = None
        return False

# 断开MAVLink连接
def disconnect_mavlink():
    if st.session_state.mav_conn is not None:
        st.session_state.mav_conn.close()
        st.session_state.mav_conn = None
        add_gcs_obc_fcu_log("MAVLink通信链路已断开")

# 读取并缓存MAVLink报文（支持多种消息）
def fetch_mavlink_msg(msg_name=None, timeout=0.1):
    conn = st.session_state.mav_conn
    if conn is None:
        return None
    try:
        if msg_name is None:
            msg = conn.recv_match(blocking=False)
        else:
            msg = conn.recv_match(type=msg_name, blocking=False, timeout=timeout)
        if msg:
            st.session_state.mav_msg_cache[msg.get_type()] = msg
        return msg
    except:
        return None

# 读取缓存内指定类型报文
def get_cached_mav_msg(msg_type):
    return st.session_state.mav_msg_cache.get(msg_type, None)

# ==================== GCJ02 <-> WGS84 坐标转换 ====================
def gcj02_to_wgs84(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat

def wgs84_to_gcj02(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat

def transform_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def out_of_china(lng, lat):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)

# ==================== 通用几何辅助函数 ====================
def distance_deg(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def distance_m(p1, p2):
    return distance_deg(p1, p2) * DEG_TO_M

def calc_path_total_m(path):
    total = 0.0
    for i in range(len(path)-1):
        total += distance_m(path[i], path[i+1])
    return round(total, 1)

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(p1, p2, p3, p4):
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def line_intersects_polygon(p1, p2, polygon):
    if point_in_polygon(p1, polygon) or point_in_polygon(p2, polygon):
        return True
    n = len(polygon)
    for i in range(n):
        p3 = polygon[i]
        p4 = polygon[(i + 1) % n]
        if segments_intersect(p1, p2, p3, p4):
            if not (p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4):
                return True
    return False

def simplify_path_by_distance(points, min_dist_deg=0.0003):
    if len(points) <= 2:
        return points
    new_path = [points[0]]
    last = points[0]
    for p in points[1:]:
        if distance_deg(last, p) >= min_dist_deg:
            new_path.append(p)
            last = p
    if new_path[-1] != points[-1]:
        new_path.append(points[-1])
    return new_path

# ==================== Catmull-Rom 平滑曲线 ====================
def catmull_rom_spline(points, num_points=6):
    if len(points) < 2:
        return points
    if len(points) == 2:
        return [points[0], points[1]]
    extended = [points[0]] + points + [points[-1]]
    spline_points = []
    for i in range(len(extended)-3):
        p0, p1, p2, p3 = extended[i], extended[i+1], extended[i+2], extended[i+3]
        for t in np.linspace(0, 1, num_points):
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                2 * p1[0] + (-p0[0] + p2[0]) * t +
                (2*p0[0] -5*p1[0] +4*p2[0] -p3[0])*t2 +
                (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3
            )
            y = 0.5 * (
                2 * p1[1] + (-p0[1] + p2[1]) * t +
                (2*p0[1] -5*p1[1] +4*p2[1] -p3[1])*t2 +
                (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3
            )
            spline_points.append([x, y])
    unique_points = []
    seen = set()
    for p in spline_points:
        key = (round(p[0],8), round(p[1],8))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    full_spline = [points[0]] + unique_points + [points[-1]]
    return simplify_path_by_distance(full_spline, 0.0003)

# ==================== 障碍物高度阻挡判断 ====================
def is_obstacle_blocking(obs, flight_height):
    obs_height = obs.get('height', 20)
    return flight_height < obs_height

def is_path_blocked(p1, p2, obstacles_gcj, flight_height):
    for obs in obstacles_gcj:
        if is_obstacle_blocking(obs, flight_height):
            coords = obs.get('polygon', [])
            if coords and len(coords)>=3:
                if line_intersects_polygon(p1, p2, coords):
                    return True
    return False

# ==================== 修复：单侧平行偏移绕行（左右向量纠正） ====================
def generate_side_bypass_path(start, end, obstacles_gcj, flight_height, safe_radius, side='left'):
    block_obs = [obs for obs in obstacles_gcj if is_obstacle_blocking(obs, flight_height)]
    if not block_obs:
        return None
    safe_radius_deg = safe_radius / DEG_TO_M
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-10:
        return None
    ux = dx / length
    uy = dy / length

    # 左右绕行向量修复
    if side == 'left':
        perp_x = -uy
        perp_y = ux
    else:
        perp_x = uy
        perp_y = -ux

    all_centers = []
    for obs in block_obs:
        poly = obs["polygon"]
        if len(poly)>=3:
            cx = sum(p[0] for p in poly)/len(poly)
            cy = sum(p[1] for p in poly)/len(poly)
            all_centers.append([cx, cy])
    if not all_centers:
        return None
    avg_cx = sum(c[0] for c in all_centers)/len(all_centers)
    avg_cy = sum(c[1] for c in all_centers)/len(all_centers)

    max_dist_to_center = 0
    for obs in block_obs:
        poly = obs["polygon"]
        for p in poly:
            dist = distance_deg([avg_cx, avg_cy], p)
            if dist > max_dist_to_center:
                max_dist_to_center = dist
    offset_distance = max_dist_to_center + safe_radius_deg * 3
    offset_point = [avg_cx + perp_x*offset_distance, avg_cy + perp_y*offset_distance]
    path = [start, offset_point, end]
    collision = False
    for i in range(len(path)-1):
        if is_path_blocked(path[i], path[i+1], obstacles_gcj, flight_height):
            collision = True
            break
    if not collision:
        smooth = catmull_rom_spline(path, 8)
        return simplify_path_by_distance(smooth)
    # 放大偏移重试
    for scale in [4,5,6,7,8,10]:
        offset_distance = max_dist_to_center + safe_radius_deg * scale
        offset_point = [avg_cx + perp_x*offset_distance, avg_cy + perp_y*offset_distance]
        path = [start, offset_point, end]
        collision = False
        for i in range(len(path)-1):
            if is_path_blocked(path[i], path[i+1], obstacles_gcj, flight_height):
                collision = True
                break
        if not collision:
            smooth = catmull_rom_spline(path,8)
            return simplify_path_by_distance(smooth)
    return None

# ==================== A* 路径规划 ====================
def astar_path(start, end, obstacles_gcj, flight_height, safe_radius):
    nodes = [start, end]
    safety = safe_radius / DEG_TO_M * 2.0
    for obs in obstacles_gcj:
        if not is_obstacle_blocking(obs, flight_height):
            continue
        poly = obs.get('polygon', [])
        if len(poly)<3:
            continue
        for i,(x,y) in enumerate(poly):
            prev_i = (i-1) % len(poly)
            prev = poly[prev_i]
            next_i = (i+1) % len(poly)
            next_p = poly[next_i]
            dx1 = -(y - prev[1])
            dy1 = x - prev[0]
            l1 = math.hypot(dx1, dy1)
            if l1>1e-8:
                dx1 /= l1
                dy1 /= l1
            nx1 = x + dx1*safety
            ny1 = y + dy1*safety
            dx2 = -(next_p[1]-y)
            dy2 = next_p[0]-x
            l2 = math.hypot(dx2, dy2)
            if l2>1e-8:
                dx2 /= l2
                dy2 /= l2
            nx2 = x + dx2*safety
            ny2 = y + dy2*safety
            nodes.append([nx1, ny1])
            nodes.append([nx2, ny2])
    unique_nodes = []
    for n in nodes:
        exist = False
        for u in unique_nodes:
            if abs(n[0]-u[0])<1e-6 and abs(n[1]-u[1])<1e-6:
                exist=True
                break
        if not exist:
            unique_nodes.append(n)
    graph = {i:[] for i in range(len(unique_nodes))}
    for i in range(len(unique_nodes)):
        for j in range(len(unique_nodes)):
            if i==j:
                continue
            if not is_path_blocked(unique_nodes[i], unique_nodes[j], obstacles_gcj, flight_height):
                graph[i].append((j, distance_deg(unique_nodes[i], unique_nodes[j])))
    start_idx, end_idx = -1, -1
    for i,n in enumerate(unique_nodes):
        if abs(n[0]-start[0])<1e-6 and abs(n[1]-start[1])<1e-6:
            start_idx = i
        if abs(n[0]-end[0])<1e-6 and abs(n[1]-end[1])<1e-6:
            end_idx = i
    if start_idx == -1 or end_idx == -1:
        return simplify_path_by_distance([start, end])
    open_heap = []
    heapq.heappush(open_heap, (0, start_idx))
    came_from = {}
    g_score = {i:float('inf') for i in range(len(unique_nodes))}
    g_score[start_idx] = 0
    f_score = {i:float('inf') for i in range(len(unique_nodes))}
    f_score[start_idx] = distance_deg(unique_nodes[start_idx], unique_nodes[end_idx])
    while open_heap:
        cur_f, cur = heapq.heappop(open_heap)
        if cur == end_idx:
            path = []
            while cur in came_from:
                path.append(unique_nodes[cur])
                cur = came_from[cur]
            path.append(unique_nodes[start_idx])
            path.reverse()
            smooth = catmull_rom_spline(path,5)
            return simplify_path_by_distance(smooth)
        for neighbor, w in graph[cur]:
            new_g = g_score[cur] + w
            if new_g < g_score[neighbor]:
                came_from[neighbor] = cur
                g_score[neighbor] = new_g
                f_score[neighbor] = new_g + distance_deg(unique_nodes[neighbor], unique_nodes[end_idx])
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
    return simplify_path_by_distance([start, end])

# ==================== 路径规划入口函数 ====================
def create_avoidance_path(start, end, obstacles_gcj, flight_height, safe_radius, strategy):
    straight_blocked = is_path_blocked(start, end, obstacles_gcj, flight_height)
    if not straight_blocked:
        path = simplify_path_by_distance([start, end])
        add_gcs_obc_fcu_log(f"航线规划完成 | 类型:直线 | 航点数:{len(path)} | 路径长度:{calc_path_total_m(path)}m")
        return path
    if strategy == 'left':
        add_gcs_obc_fcu_log(f"开始航线规划 | 类型:向左绕行 | 飞行高度:{flight_height}m")
        p = generate_side_bypass_path(start, end, obstacles_gcj, flight_height, safe_radius, 'left')
        if p and len(p)>=2:
            add_gcs_obc_fcu_log(f"航线规划完成 | 向左绕行成功 | 航点数:{len(p)} | 长度:{calc_path_total_m(p)}m")
            return p
        else:
            add_gcs_obc_fcu_log("左绕行失败，降级A*")
            ast_p = astar_path(start, end, obstacles_gcj, flight_height, safe_radius)
            add_gcs_obc_fcu_log(f"A*规划完成 | 航点数:{len(ast_p)} | 长度:{calc_path_total_m(ast_p)}m")
            return ast_p
    elif strategy == 'right':
        add_gcs_obc_fcu_log(f"开始航线规划 | 类型:向右绕行 | 飞行高度:{flight_height}m")
        p = generate_side_bypass_path(start, end, obstacles_gcj, flight_height, safe_radius, 'right')
        if p and len(p)>=2:
            add_gcs_obc_fcu_log(f"航线规划完成 | 向右绕行成功 | 航点数:{len(p)} | 长度:{calc_path_total_m(p)}m")
            return p
        else:
            add_gcs_obc_fcu_log("右绕行失败，降级A*")
            ast_p = astar_path(start, end, obstacles_gcj, flight_height, safe_radius)
            add_gcs_obc_fcu_log(f"A*规划完成 | 航点数:{len(ast_p)} | 长度:{calc_path_total_m(ast_p)}m")
            return ast_p
    else:
        obs_count = len([o for o in obstacles_gcj if is_obstacle_blocking(o, flight_height)])
        add_gcs_obc_fcu_log(f"开始A*规划 | 阻挡障碍物:{obs_count}")
        ast_p = astar_path(start, end, obstacles_gcj, flight_height, safe_radius)
        add_gcs_obc_fcu_log(f"A*规划完成 | 航点数:{len(ast_p)} | 长度:{calc_path_total_m(ast_p)}m")
        return ast_p

# ==================== 通信日志工具 ====================
def init_comm_log():
    if "gcs2fcu_log" not in st.session_state:
        st.session_state.gcs2fcu_log = []
    if "fcu2gcs_log" not in st.session_state:
        st.session_state.fcu2gcs_log = []

def add_gcs_obc_fcu_log(msg):
    init_comm_log()
    t_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.gcs2fcu_log.append(f"[{t_str}] ✅ {msg}")
    if len(st.session_state.gcs2fcu_log) > 30:
        st.session_state.gcs2fcu_log.pop(0)

def add_fcu_obc_gcs_log(msg):
    init_comm_log()
    t_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.fcu2gcs_log.append(f"[{t_str}] {msg}")
    if len(st.session_state.fcu2gcs_log) > 30:
        st.session_state.fcu2gcs_log.pop(0)

# ==================== 障碍物缓存操作 ====================
def save_obstacles_to_cache():
    st.session_state.saved_obstacles = copy.deepcopy(st.session_state.obstacles_gcj)
    st.success(f"缓存已保存 {len(st.session_state.obstacles_gcj)} 个障碍物")

def load_obstacles_from_cache():
    if "saved_obstacles" not in st.session_state or not st.session_state.saved_obstacles:
        st.warning("缓存无数据")
        return False
    st.session_state.obstacles_gcj = copy.deepcopy(st.session_state.saved_obstacles)
    st.success(f"加载缓存 {len(st.session_state.obstacles_gcj)} 个障碍物")
    return True

# ==================== 障碍物JSON导入导出 ====================
def export_obstacles_json():
    buf = io.StringIO()
    json.dump(st.session_state.obstacles_gcj, buf, ensure_ascii=False, indent=2)
    return buf.getvalue()

def import_obstacles_json(json_text):
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            st.session_state.obstacles_gcj = data
            st.success("JSON导入障碍物成功")
            st.rerun()
        else:
            st.error("JSON格式错误，需要障碍物数组")
    except Exception as e:
        st.error(f"解析失败: {str(e)}")

# ==================== 飞行心跳仿真类（新增MAV真实数据兼容） ====================
class HeartbeatSimulator:
    def __init__(self, start_point_gcj):
        self.history = []
        self.current_pos = start_point_gcj.copy()
        self.path = [start_point_gcj.copy()]
        self.path_index = 0
        self.simulating = False
        self.paused = False
        self.flight_altitude = 50
        self.speed = 50
        self.progress = 0.0
        self.total_distance = 0.0
        self.distance_traveled = 0.0
        self.start_time = None
        self.wp_logged = set()

    def set_path(self, path, altitude=50, speed=50):
        self.path = path
        self.path_index = 0
        self.current_pos = path[0].copy()
        self.flight_altitude = altitude
        self.speed = speed
        self.simulating = True
        self.paused = False
        self.progress = 0.0
        self.distance_traveled = 0.0
        self.total_distance = 0.0
        self.start_time = datetime.now()
        self.wp_logged = set()
        add_fcu_obc_gcs_log("FCU→OBC→GCS: ACK | Mode: AUTO")
        for i in range(len(path)-1):
            self.total_distance += distance_deg(path[i], path[i+1])

    def pause(self):
        self.paused = True
    def resume(self):
        self.paused = False
    def stop(self):
        self.simulating = False
    def reset(self):
        self.path_index = 0
        self.current_pos = self.path[0].copy()
        self.progress = 0.0
        self.distance_traveled = 0.0
        self.start_time = None
        self.history = []
        self.wp_logged = set()

    def update_and_generate(self):
        # 优先读取真实MAVLink数据
        real_pos_msg = get_cached_mav_msg("GLOBAL_POSITION_INT")
        real_status = get_cached_mav_msg("SYS_STATUS")
        real_att = get_cached_mav_msg("ATTITUDE")
        use_real_data = st.session_state.mav_conn is not None and real_pos_msg is not None

        if use_real_data:
            # 解析真实GPS坐标（WGS84，转GCJ02）
            raw_lat = real_pos_msg.lat / 1e7
            raw_lng = real_pos_msg.lon / 1e7
            gcj_lng, gcj_lat = wgs84_to_gcj02(raw_lng, raw_lat)
            self.current_pos = [gcj_lng, gcj_lat]
            altitude = real_pos_msg.relative_alt / 1000
            speed = real_pos_msg.vx / 100
            batt_volt = real_status.voltage_battery / 1000 if real_status else 12.5
            sat_cnt = real_pos_msg.satellites_visible
            progress = self.progress
            dist_travel = self.distance_traveled
            total_dist = self.total_distance
            elapsed = int((datetime.now()-self.start_time).total_seconds()) if self.start_time else 0
            rem_dist_m = max((total_dist - dist_travel) * DEG_TO_M, 0)
            rem_time = int(rem_dist_m / speed) if speed>0 else 0
            battery = int(batt_volt / 12.8 * 100)
            data = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "lng": gcj_lng,
                "lat": gcj_lat,
                "altitude": round(altitude,1),
                "voltage": round(batt_volt,1),
                "satellites": sat_cnt,
                "speed": round(speed,1),
                "progress": progress,
                "distance_traveled": dist_travel,
                "total_distance": total_dist,
                "simulating": self.simulating,
                "paused": self.paused,
                "elapsed_time": elapsed,
                "remaining_distance": round(rem_dist_m, 1),
                "remaining_time": rem_time,
                "battery": battery,
                "current_waypoint": self.path_index + 1,
                "total_waypoints": len(self.path)
            }
        else:
            # 原有模拟数据逻辑（无飞控连接时 fallback）
            if not self.simulating or self.paused or self.path_index >= len(self.path)-1:
                self.simulating = False
                self.progress = 1.0
            else:
                target = self.path[self.path_index+1]
                dx = target[0] - self.current_pos[0]
                dy = target[1] - self.current_pos[1]
                dist_target = math.hypot(dx, dy)
                step = 0.00008 + (self.speed/100)*0.0003
                if dist_target < step:
                    self.distance_traveled += dist_target
                    self.current_pos = target.copy()
                    wp_idx = self.path_index + 1
                    if wp_idx not in self.wp_logged:
                        add_fcu_obc_gcs_log(f"FCU→OBC→GCS: WP_REACHED #{wp_idx}")
                        self.wp_logged.add(wp_idx)
                    self.path_index += 1
                else:
                    ratio = step / dist_target
                    self.current_pos[0] += dx * ratio
                    self.current_pos[1] += dy * ratio
                    self.distance_traveled += step
                if self.total_distance > 0:
                    self.progress = min(1.0, self.distance_traveled / self.total_distance)
                if self.path_index >= len(self.path)-1:
                    self.simulating = False
                    self.progress = 1.0
                    add_fcu_obc_gcs_log("FCU→OBC→GCS: MISSION_COMPLETE")
            altitude = self.flight_altitude + random.randint(-5,5) if self.simulating else random.randint(0,10)
            speed_display = round(self.speed * 0.1, 1) if self.simulating and not self.paused else 0
            elapsed_seconds = int((datetime.now()-self.start_time).total_seconds()) if self.start_time else 0
            rem_dist_m = (self.total_distance - self.distance_traveled) * DEG_TO_M
            rem_time = int(rem_dist_m / speed_display) if speed_display>0 else 0
            battery = max(0, round(100 - (elapsed_seconds / 600)*4)) if self.simulating else 96
            data = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "lng": self.current_pos[0],
                "lat": self.current_pos[1],
                "altitude": altitude,
                "voltage": round(random.uniform(11.5,12.8),1),
                "satellites": random.randint(8,14),
                "speed": speed_display,
                "progress": self.progress,
                "distance_traveled": self.distance_traveled,
                "total_distance": self.total_distance,
                "simulating": self.simulating,
                "paused": self.paused,
                "elapsed_time": elapsed_seconds,
                "remaining_distance": round(rem_dist_m, 1),
                "remaining_time": rem_time,
                "battery": int(battery),
                "current_waypoint": self.path_index + 1,
                "total_waypoints": len(self.path)
            }
        self.history.insert(0, data)
        if len(self.history) > 200:
            self.history.pop()
        return data

# ==================== 地图渲染函数 ====================
def create_planning_map(center_gcj, points_gcj, obstacles_gcj, flight_history=None, planned_path=None, map_type="satellite", straight_blocked=True, safe_radius=5, enable_draw=True):
    if map_type == "satellite":
        tiles = ARCGIS_SATELLITE_URL
        attr = "ArcGIS卫星影像"
    else:
        tiles = AMAP_VECTOR_URL
        attr = "高德矢量地图"
    m = folium.Map(location=[center_gcj[1], center_gcj[0]], zoom_start=16, tiles=tiles, attr=attr)
    if enable_draw:
        draw = plugins.Draw(
            export=True, position='topleft',
            draw_options={'polygon': {'allowIntersection':False, 'showArea':True, 'color':'#ff0000','fillColor':'#ff0000','fillOpacity':0.4},
                          'polyline':False, 'rectangle':False, 'circle':False, 'marker':False, 'circlemarker':False},
            edit_options={'edit':True, 'remove':True}
        )
        m.add_child(draw)
    safe_offset_deg = safe_radius / DEG_TO_M
    for i, obs in enumerate(obstacles_gcj):
        poly = obs.get('polygon', [])
        if len(poly)<3:
            continue
        for (x,y) in poly:
            for angle in range(0,360,30):
                rad = math.radians(angle)
                cx = x + math.cos(rad)*safe_offset_deg
                cy = y + math.sin(rad)*safe_offset_deg
                folium.CircleMarker([cy,cx], radius=1.8, color='#00ccff', fill=True, fill_color='#00ccff', fill_opacity=0.7).add_to(m)
        poly_coords = [[c[1], c[0]] for c in poly]
        popup = f"🚧 {obs.get('name',f'障碍物{i+1}')}\n高度:{obs.get('height',20)}m"
        folium.Polygon(poly_coords, color="red", weight=3, fill=True, fill_color="red", fill_opacity=0.4, popup=popup).add_to(m)
    if points_gcj.get('A'):
        folium.Marker([points_gcj['A'][1], points_gcj['A'][0]], popup="🟢起点A", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    if points_gcj.get('B'):
        folium.Marker([points_gcj['B'][1], points_gcj['B'][0]], popup="🔴终点B", icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(m)
    if planned_path and len(planned_path)>1:
        path_loc = [[p[1], p[0]] for p in planned_path]
        folium.PolyLine(path_loc, color="green", weight=5, opacity=0.9, popup="✈️智能避障航线").add_to(m)
        for p in planned_path:
            folium.CircleMarker([p[1], p[0]], radius=3, color="green", fill=True, fill_color="white").add_to(m)
    if points_gcj.get('A') and points_gcj.get('B'):
        line_color = "blue" if not straight_blocked else "gray"
        dash_pop = "直线畅通" if not straight_blocked else "⚠️直线被阻挡"
        folium.PolyLine([[points_gcj['A'][1], points_gcj['A'][0]], [points_gcj['B'][1], points_gcj['B'][0]]], color=line_color, weight=2, opacity=0.5, dash_array="5,5", popup=dash_pop).add_to(m)
    if flight_history and len(flight_history)>1:
        trail = [[p[1], p[0]] for p in flight_history]
        folium.PolyLine(trail, color="orange", weight=2, opacity=0.6, popup="实时飞行轨迹").add_to(m)
    return m

# ==================== 主程序入口 ====================
def main():
    init_comm_log()
    init_mavlink_state() # 初始化MAV全局缓存
    st.title("🏫 无人机地面站系统 - 平行偏移绕行")
    st.markdown("---")
    # SessionState初始化
    init_vars = {
        "points_gcj": {'A':DEFAULT_A_GCJ.copy(), 'B':DEFAULT_B_GCJ.copy()},
        "obstacles_gcj": [],
        "saved_obstacles": [],
        "heartbeat_sim": HeartbeatSimulator(DEFAULT_A_GCJ.copy()),
        "last_hb_time": time.time(),
        "simulation_running": False,
        "flight_altitude": 50,
        "flight_history": [],
        "planned_path": None,
        "pending_polygon": None,
        "pending_height": 20
    }
    for k,v in init_vars.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # 侧边栏
    st.sidebar.title("🎛️ 导航菜单")
    page = st.sidebar.radio("功能模块", ["🗺️ 航线规划", "📡 飞行监控", "🚧 障碍物管理"])
    map_type_choice = st.sidebar.radio("地图类型", ["卫星影像", "矢量街道"], index=1)
    map_type = "satellite" if map_type_choice == "卫星影像" else "vector"
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 无人机参数")
    drone_speed = st.sidebar.slider("飞行速度系数", min_value=10, max_value=100, value=50, step=5)
    safe_radius = st.sidebar.number_input("安全半径(米)", min_value=1, max_value=30, value=5, step=1)
    flight_alt = st.sidebar.number_input("飞行高度(米)", min_value=0, max_value=200, value=st.session_state.flight_altitude, step=5)
    st.session_state.flight_altitude = flight_alt
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 绕行策略")
    strategy_map = {"最佳航线 (A*)":"best", "向左绕行":"left", "向右绕行":"right"}
    strategy = st.sidebar.radio("避障方式", list(strategy_map.keys()), index=0)
    selected_strategy = strategy_map[strategy]
    st.sidebar.markdown("---")
    obs_count = len(st.session_state.obstacles_gcj)
    straight_blocked = is_path_blocked(st.session_state.points_gcj['A'], st.session_state.points_gcj['B'], st.session_state.obstacles_gcj, st.session_state.flight_altitude)
    st.sidebar.info(f"🏫校园区域\n🚧障碍物:{obs_count}\n📌直线: {'🚫阻挡' if straight_blocked else '✅畅通'}")
    if st.sidebar.button("🔄 刷新重规划", use_container_width=True):
        st.session_state.planned_path = create_avoidance_path(
            st.session_state.points_gcj['A'], st.session_state.points_gcj['B'],
            st.session_state.obstacles_gcj, flight_alt, safe_radius, selected_strategy
        )
        st.rerun()
    # ========== 页面1：航线规划 ==========
    if page == "🗺️ 航线规划":
        st.header("🗺️ 航线规划 - 智能避障")
        if straight_blocked:
            st.warning(f"⚠️直线航线被建筑阻挡！障碍物高度>{flight_alt}m")
        else:
            st.success(f"✅直线航线畅通，飞行高度{flight_alt}m")
        col_ctrl, col_map = st.columns([1,1.5])
        with col_ctrl:
            st.subheader("🎮 控制面板")
            st.markdown("#### 🟢起点A")
            a_lat = st.number_input("纬度", value=st.session_state.points_gcj['A'][1], format="%.6f", key="a_lat")
            a_lng = st.number_input("经度", value=st.session_state.points_gcj['A'][0], format="%.6f", key="a_lng")
            if st.button("📍设置A点", use_container_width=True):
                st.session_state.points_gcj['A'] = [a_lng, a_lat]
                st.session_state.planned_path = create_avoidance_path(st.session_state.points_gcj['A'], st.session_state.points_gcj['B'], st.session_state.obstacles_gcj, flight_alt, safe_radius, selected_strategy)
                st.rerun()
            st.markdown("#### 🔴终点B")
            b_lat = st.number_input("纬度", value=st.session_state.points_gcj['B'][1], format="%.6f", key="b_lat")
            b_lng = st.number_input("经度", value=st.session_state.points_gcj['B'][0], format="%.6f", key="b_lng")
            if st.button("📍设置B点", use_container_width=True):
                st.session_state.points_gcj['B'] = [b_lng, b_lat]
                st.session_state.planned_path = create_avoidance_path(st.session_state.points_gcj['A'], st.session_state.points_gcj['B'], st.session_state.obstacles_gcj, flight_alt, safe_radius, selected_strategy)
                st.rerun()
            st.markdown("#### 🏗️新障碍物高度")
            new_obs_h = st.number_input("高度(米)", min_value=1, max_value=200, value=st.session_state.pending_height, step=5)
            st.session_state.pending_height = new_obs_h
            if st.button("➕添加障碍物（地图圈选）", use_container_width=True):
                if st.session_state.pending_polygon and len(st.session_state.pending_polygon)>=3:
                    st.session_state.obstacles_gcj.append({
                        "name": f"建筑物{len(st.session_state.obstacles_gcj)+1}",
                        "polygon": st.session_state.pending_polygon,
                        "height": st.session_state.pending_height
                    })
                    st.success("障碍物添加完成，自动重规划航线")
                    st.session_state.pending_polygon = None
                    st.session_state.planned_path = create_avoidance_path(st.session_state.points_gcj['A'], st.session_state.points_gcj['B'], st.session_state.obstacles_gcj, flight_alt, safe_radius, selected_strategy)
                    st.rerun()
                else:
                    st.warning("请先在地图绘制多边形障碍物")
            if st.button("🔄重新规划路径", use_container_width=True):
                st.session_state.planned_path = create_avoidance_path(st.session_state.points_gcj['A'], st.session_state.points_gcj['B'], st.session_state.obstacles_gcj, flight_alt, safe_radius, selected_strategy)
                st.rerun()
            st.markdown("#### ✈️飞行仿真控制")
            c_start, c_stop = st.columns(2)
            with c_start:
                if st.button("▶️开始飞行", use_container_width=True):
                    path = st.session_state.planned_path or [st.session_state.points_gcj['A'], st.session_state.points_gcj['B']]
                    st.session_state.heartbeat_sim.set_path(path, flight_alt, drone_speed)
                    st.session_state.simulation_running = True
                    st.session_state.flight_history = []
                    st.success("仿真任务启动")
            with c_stop:
                if st.button("⏹️停止飞行", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.session_state.heartbeat_sim.stop()
        with col_map:
            st.subheader("🗺️规划地图")
            center = st.session_state.points_gcj['A']
            if st.session_state.planned_path is None:
                st.session_state.planned_path = create_avoidance_path(
                    st.session_state.points_gcj['A'], st.session_state.points_gcj['B'],
                    st.session_state.obstacles_gcj, flight_alt, safe_radius, selected_strategy
                )
            m = create_planning_map(center, st.session_state.points_gcj, st.session_state.obstacles_gcj, st.session_state.flight_history, st.session_state.planned_path, map_type, straight_blocked, safe_radius, enable_draw=True)
            map_out = st_folium(m, width=720, height=560, returned_objects=["last_active_drawing"])
            if map_out and map_out.get("last_active_drawing"):
                draw_data = map_out["last_active_drawing"]
                if draw_data["geometry"]["type"] == "Polygon":
                    coords_raw = draw_data["geometry"]["coordinates"][0]
                    poly_gcj = [[p[0], p[1]] for p in coords_raw[:-1]]
                    if len(poly_gcj)>=3:
                        st.session_state.pending_polygon = poly_gcj
                        st.success("已捕获多边形障碍物轮廓")
    # ========== 页面2：飞行监控（MAVLink通信接口完整面板） ==========
    elif page == "📡 飞行监控":
        st.header("🛸 飞行实时画面 - 任务执行监控")
        # MAVLink通信接口设置面板（报告图3.20）
        with st.expander("📶 MAVLink通信接口设置（适配PX4 SITL UDP 14550）", expanded=True):
            conn_col1, conn_col2, conn_col3 = st.columns([2,1,1])
            udp_ip = conn_col1.text_input("飞控UDP地址", value="127.0.0.1")
            udp_port = conn_col2.number_input("端口号", value=14550, min_value=1000, max_value=60000)
            conn_status = st.session_state.mav_conn is not None
            if conn_status:
                conn_col3.success("✅ 链路已连接")
            else:
                conn_col3.error("❌ 未连接")
            btn_conn, btn_dis = st.columns(2)
            with btn_conn:
                if st.button("建立MAVLink连接", use_container_width=True):
                    if connect_mavlink(udp_ip, udp_port):
                        st.rerun()
            with btn_dis:
                if st.button("断开通信链路", use_container_width=True):
                    disconnect_mavlink()
                    st.rerun()
            # 报文类型下拉选择
            msg_type_list = ["HEARTBEAT","SYS_STATUS","VFR_HUD","ATTITUDE","GLOBAL_POSITION_INT","POWER_STATUS"]
            sel_msg = st.selectbox("选择查看MAV报文类型", msg_type_list, index=0)
            st.session_state.mav_msg_type = sel_msg
            # 实时抓取报文
            fetch_mavlink_msg()
            show_msg = get_cached_mav_msg(sel_msg)
            if show_msg:
                st.code(f"【{sel_msg}】完整报文字段\n{show_msg.to_dict()}", language="json")
            else:
                st.info("暂无该类型报文数据，启动SITL飞控后自动刷新")

        ctrl_row = st.columns([3,1])
        with ctrl_row[0]:
            btn_start, btn_pause, btn_stop, btn_reset = st.columns(4)
            with btn_start:
                if st.button("开始任务", type="primary", use_container_width=True):
                    path = st.session_state.planned_path or [st.session_state.points_gcj['A'], st.session_state.points_gcj['B']]
                    st.session_state.heartbeat_sim.set_path(path, flight_alt, drone_speed)
                    st.session_state.simulation_running = True
                    st.session_state.flight_history = []
            with btn_pause:
                if st.button("暂停", use_container_width=True):
                    st.session_state.heartbeat_sim.pause()
            with btn_stop:
                if st.button("停止", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.session_state.heartbeat_sim.stop()
            with btn_reset:
                if st.button("重置", use_container_width=True):
                    st.session_state.heartbeat_sim.reset()
                    st.session_state.simulation_running = False
                    st.session_state.flight_history = []
        with ctrl_row[1]:
            run_status = "运行中" if (st.session_state.simulation_running and not st.session_state.heartbeat_sim.paused) else "已暂停"
            st.info(f"仿真状态：{run_status}")
        # 3秒自动刷新读取MAV数据
        now_time = time.time()
        refresh_interval = 3.0
        auto_refresh = False
        if st.session_state.simulation_running or st.session_state.mav_conn is not None:
            if now_time - st.session_state.last_hb_time >= refresh_interval:
                sim_data = st.session_state.heartbeat_sim.update_and_generate()
                pos = [sim_data["lng"], sim_data["lat"]]
                st.session_state.flight_history.append(pos)
                if len(st.session_state.flight_history) > 200:
                    st.session_state.flight_history.pop(0)
                st.session_state.last_hb_time = now_time
                auto_refresh = True
        if auto_refresh:
            st.rerun()
        # 飞行数据仪表盘（图3.21/3.22，兼容真实MAV/模拟数据）
        if st.session_state.heartbeat_sim.history:
            latest = st.session_state.heartbeat_sim.history[0]
            metric_cols = st.columns(6)
            metric_cols[0].metric("当前航点", f"{latest['current_waypoint']}/{latest['total_waypoints']}")
            metric_cols[1].metric("飞行速度", f"{latest['speed']} m/s")
            metric_cols[2].metric("已用时间", str(timedelta(seconds=latest['elapsed_time'])))
            metric_cols[3].metric("剩余距离", f"{latest['remaining_distance']} m")
            metric_cols[4].metric("预计到达", str(timedelta(seconds=latest['remaining_time'])) if latest['remaining_time']>0 else "00:00")
            metric_cols[5].metric("模拟电量", f"{latest['battery']}%")
            st.progress(latest["progress"], text=f"任务进度 {latest['progress']*100:.0f}%")
            st.markdown("---")
            map_col, log_col = st.columns([2,1])
            with map_col:
                st.subheader("实时飞行地图")
                m = create_planning_map(st.session_state.points_gcj['A'], st.session_state.points_gcj, st.session_state.obstacles_gcj, st.session_state.flight_history, st.session_state.planned_path, map_type, straight_blocked, safe_radius, enable_draw=False)
                folium_static(m, width=620, height=420)
            with log_col:
                st.subheader("📡 通信链路拓扑（图3.23）")
                topo_html = '''
                <div style="display:flex; justify-content:space-around; text-align:center; margin-top:10px;">
                    <div style="width:28%; padding:12px; background:#e3f2fd; border:2px solid #1976d2; border-radius:8px;">
                        <div style="font-weight:bold; font-size:16px; color:#1976d2;">GCS地面站</div>
                        <div style="font-size:12px; color:gray;">192.168.1.100</div>
                        <div style="color:green;">✅在线</div>
                    </div>
                    <div>⬇️UDP 14550⬆️</div>
                    <div style="width:28%; padding:12px; background:#fff8e1; border:2px solid #f57c00; border-radius:8px;">
                        <div style="font-weight:bold; font-size:16px; color:#f57c00;">OBC机载计算机</div>
                        <div style="font-size:12px; color:gray;">Raspberry Pi4</div>
                        <div style="color:green;">✅在线</div>
                    </div>
                    <div>⬇️MAVLink⬆️</div>
                    <div style="width:28%; padding:12px; background:#fce4ec; border:2px solid #c2185b; border-radius:8px;">
                        <div style="font-weight:bold; font-size:16px; color:#c2185b;">FCU飞控</div>
                        <div style="font-size:12px; color:gray;">PX4/ArduPilot</div>
                        <div style="color:green;">✅在线</div>
                    </div>
                </div>
                <div style="margin-top:15px; padding:8px; background:#f5f5f5; border-radius:6px;">
                链路统计：平均延迟~25ms，丢包率0.1%
                </div>
                '''
                st.markdown(topo_html, unsafe_allow_html=True)
                tab_down, tab_up = st.tabs(["下发日志 GCS→FCU", "回传日志 FCU→GCS"])
                with tab_down:
                    log_text = "\n".join(st.session_state.gcs2fcu_log) if st.session_state.gcs2fcu_log else "暂无下发日志"
                    st.text_area("", log_text, height=220)
                with tab_up:
                    log_text = "\n".join(st.session_state.fcu2gcs_log) if st.session_state.fcu2gcs_log else "暂无回传日志"
                    st.text_area("", log_text, height=220)
    # ========== 页面3：障碍物管理 ==========
    elif page == "🚧 障碍物管理":
        st.header("🚧 障碍物管理面板")
        st.info(f"当前障碍物总数：{len(st.session_state.obstacles_gcj)}")
        col_list, col_opt = st.columns([1,1.5])
        with col_list:
            st.subheader("障碍物列表")
            if len(st.session_state.obstacles_gcj) == 0:
                st.write("暂无障碍物，请前往航线规划页面绘制添加")
            else:
                for idx, obs in enumerate(st.session_state.obstacles_gcj):
                    name_col, h_col, del_col = st.columns([2,1,1])
                    name_col.write(f"🏢 {obs['name']}")
                    h_col.write(f"{obs['height']} m")
                    if del_col.button("删除", key=f"del_obs_{idx}"):
                        st.session_state.obstacles_gcj.pop(idx)
                        st.rerun()
            st.markdown("### 缓存操作")
            save_cache, load_cache = st.columns(2)
            with save_cache:
                st.button("💾保存到缓存", on_click=save_obstacles_to_cache, use_container_width=True)
            with load_cache:
                st.button("📂加载缓存", on_click=load_obstacles_from_cache, use_container_width=True)
            if st.button("🗑️全部清空障碍物", use_container_width=True):
                st.session_state.obstacles_gcj = []
                st.rerun()
        with col_opt:
            st.subheader("JSON 文件导入 / 导出")
            json_str = export_obstacles_json()
            st.download_button(
                label="📤导出障碍物JSON文件",
                data=json_str,
                file_name="obstacles_data.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown("#### 导入JSON文件")
            upload_file = st.file_uploader("上传obstacles_data.json", type="json")
            if upload_file is not None:
                file_content = upload_file.read().decode("utf-8")
                if st.button("确认导入", use_container_width=True):
                    import_obstacles_json(file_content)

if __name__ == "__main__":
    main()
