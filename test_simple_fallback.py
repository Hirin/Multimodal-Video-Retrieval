import streamlit as st

st.title("Simple Fallback: Manual Time Display")

# Test button to open popover
if st.button("Open Video Popover"):
    with st.popover("🎥 Simple Video Player", use_container_width=True):
        st.markdown("**Simple Video Player with Manual Time Display**")
        
        frame_col, video_col = st.columns([1, 2])
        
        with frame_col:
            st.markdown("**Video Info:**")
            start_sec = 833
            
            # Simple static display
            st.markdown(f"""
            <div style="font-size: 18px; padding: 20px; background-color: #2e2e2e; border-radius: 8px; text-align: center; color: white;">
                <p><strong>Start Time:</strong></p>
                <p style="font-size: 24px; color: #00ff00; margin: 10px 0;">{start_sec:.1f}s</p>
                <p><strong>Duration:</strong> ~20 minutes</p>
                <p><strong>Video ID:</strong> Rzpw5WR7nAY</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Manual time input
            st.markdown("**Manual Time Input:**")
            manual_time = st.number_input(
                "Current Time (seconds)", 
                min_value=0.0, 
                max_value=1200.0, 
                value=float(start_sec),
                step=1.0,
                key="manual_time"
            )
            
            # Convert to mm:ss format
            minutes = int(manual_time // 60)
            seconds = int(manual_time % 60)
            st.markdown(f"**Time:** {minutes:02d}:{seconds:02d}")
            
            # Quick jump buttons
            st.markdown("**Quick Jump:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Start", key="jump_start"):
                    st.session_state.manual_time = start_sec
                    st.rerun()
            with col2:
                if st.button("+30s", key="jump_plus"):
                    st.session_state.manual_time += 30
                    st.rerun()
            with col3:
                if st.button("-30s", key="jump_minus"):
                    st.session_state.manual_time = max(start_sec, st.session_state.manual_time - 30)
                    st.rerun()
        
        with video_col:
            st.markdown("**Video Player:**")
            video_id = "Rzpw5WR7nAY"
            
            # Simple iframe without API
            st.markdown(f"""
            <iframe id="ytplayer-simple" type="text/html" width="100%" height="500"
            src="https://www.youtube.com/embed/{video_id}?start={start_sec}&enablejsapi=0"
            frameborder="0" allowfullscreen></iframe>
            """, unsafe_allow_html=True)
            
            st.markdown("**Instructions:**")
            st.markdown("- Use the number input to track time manually")
            st.markdown("- Use quick jump buttons to navigate")
            st.markdown("- This avoids all YouTube API issues")

st.markdown("---")
st.markdown("**This approach:**")
st.markdown("✅ **Always works** - No API dependencies")
st.markdown("✅ **Simple and reliable** - Just basic HTML iframe")
st.markdown("✅ **User control** - Manual time tracking")
st.markdown("❌ **Not automatic** - Requires manual input")
st.markdown("❌ **Less smooth** - No real-time updates")
