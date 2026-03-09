from playwright.sync_api import sync_playwright

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8080/")
        page.wait_for_selector("text=Start Evolution (1 Gen)")
        page.screenshot(path="verification.png")
        browser.close()

if __name__ == "__main__":
    verify()
