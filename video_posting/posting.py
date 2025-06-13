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
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"capcut_{name}_{timestamp}.png"
        print(f"Attempting to save screenshot: {filename}")
        driver.save_screenshot(filename)
        print(f"Successfully saved screenshot: {filename}")
    except Exception as e:
        print(f"Error saving screenshot {name}: {str(e)}")

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
            print(f"\nTrying button selector: {selector}")
            elements = driver.find_elements(By.XPATH, selector)
            if elements:
                print(f"Found {len(elements)} elements with selector {selector}")
                for elem in elements:
                    print(f"Element text: {elem.text}")
                    print(f"Element class: {elem.get_attribute('class')}")
                    print(f"Element role: {elem.get_attribute('role')}")
                    print(f"Element type: {elem.get_attribute('type')}")
                    print(f"Element name: {elem.get_attribute('name')}")
                    print(f"Element id: {elem.get_attribute('id')}")
                    print(f"Element is displayed: {elem.is_displayed()}")
                    print(f"Element is enabled: {elem.is_enabled()}")
                    print(f"Element location: {elem.location}")
                    print(f"Element size: {elem.size}")
                    print("---")
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
    
    print(f"\nAttempting to click button{': ' + button_text if button_text else ''}...")
    print(f"Button details before click:")
    print(f"Text: {next_button.text}")
    print(f"Class: {next_button.get_attribute('class')}")
    print(f"Role: {next_button.get_attribute('role')}")
    print(f"Type: {next_button.get_attribute('type')}")
    print(f"Name: {next_button.get_attribute('name')}")
    print(f"ID: {next_button.get_attribute('id')}")
    print(f"Is displayed: {next_button.is_displayed()}")
    print(f"Is enabled: {next_button.is_enabled()}")
    print(f"Location: {next_button.location}")
    print(f"Size: {next_button.size}")
    
    try:
        # Try JavaScript click first
        print("Attempting JavaScript click...")
        driver.execute_script("arguments[0].click();", next_button)
        print("JavaScript click successful")
    except Exception as e:
        print(f"JavaScript click failed: {str(e)}")
        print("Attempting regular click...")
        next_button.click()
        print("Regular click successful")
    
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
        print(f"\nLooking for .env file at: {env_path}")
        print(f"File exists: {os.path.exists(env_path)}")
        load_dotenv(env_path)
        email = os.getenv('CAPCUT_EMAIL')
        password = os.getenv('CAPCUT_PASSWORD')
        recovery_email = os.getenv('CAPCUT_RECOVERY_EMAIL')
        
        print(f"\nEnvironment variables loaded:")
        print(f"Email exists: {bool(email)}")
        print(f"Password exists: {bool(password)}")
        print(f"Recovery email exists: {bool(recovery_email)}")
        
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
        
        # Click Next after password
        print("\nClicking Next after password...")
        try:
            # Take screenshot before clicking Next
            save_screenshot(driver, "08a_before_password_next")
            
            # Print page source for debugging
            print("\nPage source after password entry:")
            print(driver.page_source[:1000])  # Print first 1000 characters
            
            # Click Next
            if not find_and_click_button(driver, "Next"):
                raise Exception("Failed to click Next after password")
            
            # Check if popup is still open before taking screenshot
            if popup_handle in driver.window_handles:
                try:
                    driver.switch_to.window(popup_handle)
                    save_screenshot(driver, "08b_after_password_next")
                except Exception as e:
                    print(f"Could not take screenshot after Next click: {str(e)}")
            
            # Check if popup is still open
            if popup_handle not in driver.window_handles:
                print("Popup window closed after clicking Next - this might indicate successful login")
                driver.switch_to.window(main_handle)
                print("Waiting 5 seconds for page to settle...")
                time.sleep(5)  # Wait 5 seconds after popup closes
                save_screenshot(driver, "11_final")
                return True
            
            # Wait a moment for the page to load
            time.sleep(2)
            
            # Print all text elements on the page to help debug
            print("\nAll text elements on the page:")
            elements = driver.find_elements(By.XPATH, "//*[text()]")
            for elem in elements:
                print(f"Text: {elem.text}")
                print(f"Tag: {elem.tag_name}")
                print(f"Class: {elem.get_attribute('class')}")
                print("---")
            
            # Look for recovery email input field
            print("\nLooking for recovery email input field...")
            try:
                # Try multiple selectors for the recovery email input
                recovery_selectors = [
                    "input[type='email']",
                    "input[name='recoveryEmail']",
                    "input[aria-label*='recovery']",
                    "input[placeholder*='recovery']"
                ]
                
                recovery_input = None
                for selector in recovery_selectors:
                    try:
                        print(f"Trying selector: {selector}")
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            print(f"Found {len(elements)} elements with selector {selector}")
                            for elem in elements:
                                print(f"Element type: {elem.get_attribute('type')}")
                                print(f"Element name: {elem.get_attribute('name')}")
                                print(f"Element class: {elem.get_attribute('class')}")
                                if elem.is_displayed() and elem.is_enabled():
                                    recovery_input = elem
                                    break
                        
                        if recovery_input:
                            print(f"Found recovery input using selector: {selector}")
                            break
                    except Exception as e:
                        print(f"Selector {selector} failed: {str(e)}")
                        continue
                
                if recovery_input:
                    print(f"Entering recovery email: {recovery_email}")
                    recovery_input.clear()
                    recovery_input.send_keys(recovery_email)
                    time.sleep(1)
                    save_screenshot(driver, "09_recovery_email_entered")
                    
                    # Click Next after recovery email
                    if not find_and_click_button(driver, "Next"):
                        raise Exception("Failed to click Next after recovery email")
                    save_screenshot(driver, "10_after_recovery_email_next")
                else:
                    print("Could not find recovery email input field")
                    save_screenshot(driver, "09_recovery_email_not_found")
                    
            except Exception as e:
                print(f"Error entering recovery email: {str(e)}")
                save_screenshot(driver, "09_recovery_email_error")
            
            # Check if popup is still open
            if popup_handle not in driver.window_handles:
                print("Popup window closed after recovery email - this is normal for successful login")
                # Switch back to main window
                driver.switch_to.window(main_handle)
                print("Waiting 5 seconds for page to settle...")
                time.sleep(5)  # Wait 5 seconds after popup closes
                save_screenshot(driver, "11_final")
                return True
            
        except Exception as e:
            print(f"Error after clicking Next: {str(e)}")
            # Check if popup is still open
            if popup_handle in driver.window_handles:
                save_screenshot(driver, "08c_error_after_next")
            else:
                print("Popup window closed after error - this might indicate successful login")
                driver.switch_to.window(main_handle)
                print("Waiting 5 seconds for page to settle...")
                time.sleep(5)  # Wait 5 seconds after popup closes
                save_screenshot(driver, "11_final")
                return True
        
        # Switch back to main window and wait
        print("\nSwitching back to main window...")
        driver.switch_to.window(main_handle)
        print("Waiting 30 seconds for page to settle...")
        time.sleep(30)  # Total wait time of 30 seconds
        
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
        
        print("\nLooking specifically for 'Continue with Google' button...")
        
        # First, let's print all elements with exact text 'Continue with Google'
        print("\nScanning all elements for exact text 'Continue with Google':")
        exact_text_elements = driver.find_elements(By.XPATH, "//*[text()='Continue with Google']")
        for elem in exact_text_elements:
            print(f"\nFound element with exact text:")
            print(f"Tag: {elem.tag_name}")
            print(f"Class: {elem.get_attribute('class')}")
            print(f"ID: {elem.get_attribute('id')}")
            print(f"Text: {elem.text}")
            print(f"Is displayed: {elem.is_displayed()}")
            print(f"Is enabled: {elem.is_enabled()}")
        
        # Try multiple specific selectors for Google button
        google_selectors = [
            "//div[@id='container']//*[contains(text(), 'Continue with Google')]",  # Search within container
            "//div[contains(@class, 'google_sign_in')]",
            "//div[contains(@class, 'google')]",
            "//button[contains(@class, 'google')]",
            "//a[contains(@class, 'google')]",
            "//div[contains(@class, 'sign-in')]",
            "//div[contains(@class, 'signin')]",
            "//div[contains(@class, 'login')]",
            # Search for any element containing the exact text
            "//*[contains(text(), 'Continue with Google')]",
            # Search for elements that might contain the text in a child element
            "//*[.//text()[contains(., 'Continue with Google')]]"
        ]
        
        for selector in google_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                print(f"\nFound {len(elements)} elements with selector: {selector}")
                
                for elem in elements:
                    print(f"\nExamining element:")
                    print(f"Tag: {elem.tag_name}")
                    print(f"Class: {elem.get_attribute('class')}")
                    print(f"ID: {elem.get_attribute('id')}")
                    print(f"Text: {elem.text}")
                    print(f"Is displayed: {elem.is_displayed()}")
                    print(f"Is enabled: {elem.is_enabled()}")
                    
                    # Get parent element's text as well
                    try:
                        parent = elem.find_element(By.XPATH, "..")
                        print(f"Parent text: {parent.text}")
                        print(f"Parent class: {parent.get_attribute('class')}")
                        print(f"Parent ID: {parent.get_attribute('id')}")
                    except:
                        print("Could not get parent element")
                    
                    # Get all child elements' text
                    try:
                        children = elem.find_elements(By.XPATH, ".//*")
                        print("\nChild elements:")
                        for child in children:
                            print(f"Child tag: {child.tag_name}")
                            print(f"Child text: {child.text}")
                            print(f"Child class: {child.get_attribute('class')}")
                            print("---")
                    except:
                        print("Could not get child elements")
                    
                    if elem.is_displayed() and elem.is_enabled():
                        # Try to click the element
                        try:
                            print("Attempting to click element...")
                            driver.execute_script("arguments[0].click();", elem)
                            print("Successfully clicked Google button using JavaScript")
                            save_screenshot(driver, "02_google_button_clicked")
                            
                            # Handle the popup window
                            if not handle_google_popup(driver):
                                raise Exception("Failed to handle Google sign-in popup")
                            
                            return True
                        except Exception as e:
                            print(f"JavaScript click failed: {str(e)}")
                            try:
                                elem.click()
                                print("Successfully clicked Google button using regular click")
                                save_screenshot(driver, "02_google_button_clicked")
                                
                                # Handle the popup window
                                if not handle_google_popup(driver):
                                    raise Exception("Failed to handle Google sign-in popup")
                                
                                return True
                            except Exception as e:
                                print(f"Regular click failed: {str(e)}")
            except Exception as e:
                print(f"Selector {selector} failed: {str(e)}")
                continue
        
        print("\nCould not find Google sign-in button")
        return False
        
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