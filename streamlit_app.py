# -*- coding: utf-8 -*-
"""Streamlit Community Cloud 默认入口名（streamlit_app.py）。
本项目的真正主程序是 app.py，这里只做转发，避免平台把入口写死成默认名时报错。
"""
import os
import runpy

os.chdir(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
               run_name="__main__")
