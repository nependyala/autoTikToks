"""
Script to handle video posting functionality.
"""
import os
import sys
import time
import glob
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

def cleanup_old_screenshots():
    """
    Delete all existing screenshots that start with 'capcut_'
    """
    print("Cleaning up old screenshots...")
    screenshots = glob.glob("capcut_*.png")
    for screenshot in screenshots:
        try:
            os.remove(screenshot)
            print(f"Deleted: {screenshot}")
        except Exception as e:
            print(f"Error deleting {screenshot}: {str(e)}")

def setup_driver() -> webdriver.Chrome:
    """
    Set up and return a configured Chrome WebDriver.
    
    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")  # Required for running in AWS
    chrome_options.add_argument("--disable-dev-shm-usage")  # Required for running in AWS
    chrome_options.add_argument("--window-size=1920,1080")  # Set a standard window size
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def save_screenshot(driver: webdriver.Chrome, name: str):
    """
    Save a screenshot with timestamp.
    
    Args:
        driver (webdriver.Chrome): Chrome WebDriver instance
        name (str): Name for the screenshot
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"capcut_{name}_{timestamp}.png"
    driver.save_screenshot(filename)
    print(f"Saved screenshot: {filename}")

def find_and_click_button(driver: webdriver.Chrome, button_text: str = None) -> bool:
    """
    Find and click a button using multiple selectors.
    
    Args:
        driver (webdriver.Chrome): Chrome WebDriver instance
        button_text (str): Optional text to look for in the button
        
    Returns:
        bool: True if button was found and clicked, False otherwise
    """
    next_button_selectors = [
        f"//button[contains(text(), '{button_text}')]" if button_text else None,
        "//button[@type='button']",
        "//div[@role='button']",
        "//div[contains(@class, 'button')]",
        "//div[contains(@class, 'next')]"
    ]
    
    # Remove None values from selectors
    next_button_selectors = [s for s in next_button_selectors if s]
    
    next_button = None
    for selector in next_button_selectors:
        try:
            print(f"Trying button selector: {selector}")
            elements = driver.find_elements(By.XPATH, selector)
            if elements:
                print(f"Found {len(elements)} elements with selector {selector}")
                for elem in elements:
                    print(f"Element text: {elem.text}")
                    print(f"Element class: {elem.get_attribute('class')}")
                    print(f"Element role: {elem.get_attribute('role')}")
                    if elem.is_displayed() and elem.is_enabled():
                        next_button = elem
                        break
            
            if next_button:
                print(f"Found clickable button using selector: {selector}")
                break
        except Exception as e:
            print(f"Button selector {selector} failed: {str(e)}")
            continue
    
    if not next_button:
        raise Exception(f"Could not find button{': ' + button_text if button_text else ''}")
    
    print(f"Clicking button{': ' + button_text if button_text else ''}...")
    next_button.click()
    time.sleep(2)
    return True

def handle_google_popup(driver: webdriver.Chrome) -> bool:
    """
    Handle the Google sign-in popup window.
    
    Args:
        driver (webdriver.Chrome): Chrome WebDriver instance
        
    Returns:
        bool: True if popup was handled successfully, False otherwise
    """
    try:
        print("Waiting for Google sign-in popup...")
        time.sleep(2)  # Wait for popup to open
        
        # Get all window handles
        handles = driver.window_handles
        if len(handles) < 2:
            print("No popup window found")
            return False
            
        print(f"Found {len(handles)} windows")
        
        # Store main window handle
        main_handle = handles[0]
        
        # Switch to the popup window (last handle)
        popup_handle = handles[-1]
        driver.switch_to.window(popup_handle)
        print("Switched to popup window")
        save_screenshot(driver, "03_popup_initial")
        
        # Load credentials from environment variables
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'secrets', '.env')
        load_dotenv(env_path)
        email = os.getenv('CAPCUT_EMAIL')
        password = os.getenv('CAPCUT_PASSWORD')
        recovery_email = os.getenv('CAPCUT_RECOVERY_EMAIL')
        
        if not email or not password or not recovery_email:
            raise Exception("CapCut credentials not found in environment variables")
        
        # Handle email input
        print("Looking for email input field...")
        email_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']"))
        )
        
        print(f"Entering email: {email}")
        email_input.clear()  # Clear any existing text
        email_input.send_keys(email)
        time.sleep(2)  # Wait longer for input to be processed
        save_screenshot(driver, "04_email_entered")
        
        # Click Next after email
        if not find_and_click_button(driver, "Next"):
            raise Exception("Failed to click Next after email")
        save_screenshot(driver, "05_after_email_next")
        
        # Wait for page to load and ensure we're still in popup
        time.sleep(5)  # Increased wait time
        if popup_handle not in driver.window_handles:
            print("Popup window closed unexpectedly")
            return False
        driver.switch_to.window(popup_handle)
        save_screenshot(driver, "06_password_page")
        
        # Handle password input
        print("Looking for password input field...")
        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
        )
        
        print("Entering password...")
        password_input.clear()  # Clear any existing text
        password_input.send_keys(password)
        time.sleep(2)  # Wait longer for input to be processed
        save_screenshot(driver, "07_password_entered")
        
        # Click Next after password and immediately search for recovery confirmation
        print("\nClicking Next and immediately searching for recovery confirmation...")
        try:
            # Take screenshot before clicking Next
            save_screenshot(driver, "08a_before_password_next")
            
            # Click Next
            if not find_and_click_button(driver, "Next"):
                raise Exception("Failed to click Next after password")
            
            # Take screenshot immediately after clicking Next
            save_screenshot(driver, "08b_after_password_next")
            
            # Immediately search for recovery email confirmation
            print("Searching for recovery email confirmation...")
            recovery_selectors = [
                "//div[contains(text(), 'Confirm your recovery email')]",
                "//div[contains(text(), 'recovery email')]",
                "//div[contains(text(), 'recovery')]",
                "//*[contains(text(), 'Confirm your recovery email')]",
                "//*[contains(text(), 'recovery email')]",
                "//*[contains(text(), 'recovery')]"
            ]
            
            recovery_element = None
            for selector in recovery_selectors:
                try:
                    print(f"Trying selector: {selector}")
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        print(f"Found {len(elements)} elements with selector {selector}")
                        for elem in elements:
                            print(f"Element text: {elem.text}")
                            print(f"Element tag: {elem.tag_name}")
                            print(f"Element class: {elem.get_attribute('class')}")
                            if elem.is_displayed() and elem.is_enabled():
                                recovery_element = elem
                                break
                    
                    if recovery_element:
                        print(f"Found clickable recovery element using selector: {selector}")
                        break
                except Exception as e:
                    print(f"Selector {selector} failed: {str(e)}")
                    continue
            
            if recovery_element:
                print("Clicking recovery email confirmation...")
                recovery_element.click()
                save_screenshot(driver, "08c_recovery_confirmation_clicked")
            else:
                print("Could not find recovery email confirmation element")
                save_screenshot(driver, "08c_recovery_not_found")
            
            # Check if popup is still open
            if popup_handle not in driver.window_handles:
                print("Popup window closed after recovery confirmation - this is normal for successful login")
                # Switch back to main window
                driver.switch_to.window(main_handle)
                time.sleep(10)  # Wait for page to settle
                time.sleep(2)  # Additional 2-second wait before final screenshot
                save_screenshot(driver, "11_final")
                return True
            
            # Now look for the recovery email input field
            try:
                recovery_input = WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']"))
                )
                
                print(f"Entering recovery email: {recovery_email}")
                recovery_input.clear()
                recovery_input.send_keys(recovery_email)
                time.sleep(1)
                save_screenshot(driver, "09_recovery_email_entered")
                
                # Click Next after recovery email
                if not find_and_click_button(driver, "Next"):
                    raise Exception("Failed to click Next after recovery email")
                save_screenshot(driver, "10_after_recovery_next")
            except Exception as e:
                print(f"Error handling recovery email: {str(e)}")
                print("Taking screenshot of popup window...")
                if popup_handle in driver.window_handles:
                    driver.switch_to.window(popup_handle)
                    save_screenshot(driver, "09_recovery_email_not_found")
            
        except Exception as e:
            print(f"Error after clicking Next: {str(e)}")
            # Check if popup is still open
            if popup_handle in driver.window_handles:
                save_screenshot(driver, "08d_error_after_next")
            else:
                print("Popup window closed after error - this might indicate successful login")
                driver.switch_to.window(main_handle)
                time.sleep(10)
                time.sleep(2)  # Additional 2-second wait before final screenshot
                save_screenshot(driver, "11_final")
                return True
        
        # Switch back to main window and wait
        print("\nSwitching back to main window...")
        driver.switch_to.window(main_handle)
        time.sleep(10)  # Wait 10 seconds for page to settle
        time.sleep(2)  # Additional 2-second wait before final screenshot
        
        # Take final screenshot of main window
        print("Taking final screenshot...")
        save_screenshot(driver, "11_final")
        
        return True
        
    except Exception as e:
        print(f"Error handling popup: {str(e)}")
        # Take error screenshot if possible
        try:
            save_screenshot(driver, "error_state")
        except:
            pass
        return False

def click_google_button(driver: webdriver.Chrome) -> bool:
    """
    Find and click the Continue with Google button.
    
    Args:
        driver (webdriver.Chrome): Chrome WebDriver instance
        
    Returns:
        bool: True if button was found and clicked, False otherwise
    """
    try:
        print("Navigating to CapCut login page...")
        driver.get("https://www.capcut.com/login")
        time.sleep(5)  # Wait for page to load
        save_screenshot(driver, "01_login_page")
        
        print("Looking for Google button...")
        # Try to find the specific Google button element
        selectors = [
            "//div[@class='lv_google_sign_in_btn-expand-wrapper']",
            "//div[contains(@class, 'lv_google_sign_in_btn-expand-wrapper')]",
            "//div[contains(@class, 'lv_google_sign_in_btn')]",
            "//div[contains(@class, 'google_sign_in')]"
        ]
        
        google_button = None
        for selector in selectors:
            try:
                print(f"\nTrying button selector: {selector}")
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector {selector}")
                    for elem in elements:
                        print(f"Element class: {elem.get_attribute('class')}")
                        print(f"Element text: {elem.text}")
                        print(f"Element tag: {elem.tag_name}")
                        if elem.is_displayed() and elem.is_enabled():
                            google_button = elem
                            break
                
                if google_button:
                    print(f"Found clickable Google button using selector: {selector}")
                    break
            except Exception as e:
                print(f"Button selector {selector} failed: {str(e)}")
                continue
        
        if not google_button:
            raise Exception("Could not find Google button")
        
        print("Clicking Google button...")
        google_button.click()
        save_screenshot(driver, "02_google_button_clicked")
        
        # Handle the popup window
        if not handle_google_popup(driver):
            raise Exception("Failed to handle Google sign-in popup")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting CapCut browser automation...")
    cleanup_old_screenshots()  # Clean up old screenshots before starting
    driver = setup_driver()
    
    try:
        click_google_button(driver)
    finally:
        driver.quit() 