"""
UI styling components for the Streamlit application.
"""


def create_popover_css() -> str:
    """
    Create CSS to improve popover size and appearance.

    Returns:
        CSS style string
    """
    return """
    <style>
    .stPopover {
        min-width: 450px !important;
        min-height: 100px !important;
        padding: 25px !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
    }
    .stPopover > div {
        padding: 25px !important;
        min-height: 80px !important;
    }
    .stPopover [data-testid="stRadio"] {
        margin: 10px 0 !important;
    }
    .stPopover [data-testid="stRadio"] > div {
        gap: 15px !important;
    }
    </style>
    """


def create_loading_overlay(message: str) -> str:
    """
    Create a custom loading overlay with blur background.

    Args:
        message: Loading message to display

    Returns:
        HTML string for loading overlay
    """
    return f"""
    <div style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(5px);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            text-align: center;
            color: white;
            max-width: 400px;
            animation: pulse 2s infinite;
        ">
            <div style="
                font-size: 48px;
                margin-bottom: 20px;
                animation: spin 1s linear infinite;
            ">⏳</div>
            <h3 style="margin: 0 0 10px 0; font-size: 24px; font-weight: 600;">Đang tải video...</h3>
            <p style="margin: 0; font-size: 16px; opacity: 0.9;">{message}</p>
            <div style="
                margin-top: 20px;
                width: 100%;
                height: 4px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 2px;
                overflow: hidden;
            ">
                <div style="
                    height: 100%;
                    background: linear-gradient(90deg, #00d4ff, #00ff88);
                    border-radius: 2px;
                    animation: progress 2s ease-in-out infinite;
                "></div>
            </div>
        </div>
    </div>

    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        @keyframes progress {{
            0% {{ width: 0%; }}
            50% {{ width: 70%; }}
            100% {{ width: 100%; }}
        }}
    </style>
    """


def create_dialog_loading_html(message: str) -> str:
    """
    Create loading animation HTML for dialogs.

    Args:
        message: Loading message to display

    Returns:
        HTML string for dialog loading animation
    """
    return f"""
    <div style="
        text-align: center;
        padding: 40px 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <div style="
            font-size: 60px;
            margin-bottom: 30px;
            animation: spin 1s linear infinite;
        ">⏳</div>
        <div style="font-size: 24px; font-weight: 600; color: #333; margin-bottom: 20px;">
            Đang tải video...
        </div>
        <div style="font-size: 18px; color: #666; margin-bottom: 30px;">
            {message}
        </div>
        <div style="font-size: 16px; color: #999;">
            Vui lòng đợi trong giây lát...
        </div>
        <div style="
            margin-top: 40px;
            width: 100%;
            height: 8px;
            background: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 4px;
                animation: progress 2s ease-in-out infinite;
            "></div>
        </div>
    </div>

    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes progress {{
            0% {{ width: 0%; }}
            50% {{ width: 70%; }}
            100% {{ width: 100%; }}
        }}
    </style>
    """


def create_success_html() -> str:
    """
    Create success message HTML.

    Returns:
        HTML string for success message
    """
    return """
    <div style="
        text-align: center;
        padding: 40px 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <div style="
            font-size: 60px;
            margin-bottom: 30px;
            animation: bounce 0.6s ease-in-out;
        ">✅</div>
        <div style="font-size: 24px; font-weight: 600; color: #333; margin-bottom: 20px;">
            Tải thành công!
        </div>
        <div style="font-size: 18px; color: #666;">
            Video đã được tải về thành công
        </div>
    </div>

    <style>
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }
    </style>
    """


def create_error_html() -> str:
    """
    Create error message HTML.

    Returns:
        HTML string for error message
    """
    return """
    <div style="
        text-align: center;
        padding: 40px 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <div style="
            font-size: 60px;
            margin-bottom: 30px;
            animation: shake 0.5s ease-in-out;
        ">❌</div>
        <div style="font-size: 24px; font-weight: 600; color: #333; margin-bottom: 20px;">
            Tải thất bại!
        </div>
        <div style="font-size: 18px; color: #666;">
            Có lỗi xảy ra trong quá trình tải video
        </div>
    </div>

    <style>
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
    </style>
    """
