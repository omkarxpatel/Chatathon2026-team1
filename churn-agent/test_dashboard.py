"""Run after generating demo data: python -m unittest test_dashboard -v"""

import unittest

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file("app/dashboard.py", default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def click(self, label):
        next(b for b in self.app.button if b.label == label).click().run()
        self.assertFalse(self.app.exception)

    def open_example(self, cid):
        self.app.button(key="nav-How it works").click().run()
        self.app.button(key=f"demo-{cid}").click().run()
        self.assertFalse(self.app.exception)

    def test_risk_order_pagination_and_empty_search(self):
        buttons = [b for b in self.app.button if b.key and b.key.startswith("open-")]
        self.assertEqual(buttons[0].key, "open-CUST-0037")
        self.assertEqual(len(buttons), 6)
        self.assertTrue(all(b.label == "Open review" for b in buttons))
        self.click("Next")
        self.assertEqual(self.app.session_state.queue_page, 1)
        self.app.text_input(key="customer_search").set_value("no-such-customer").run()
        self.assertEqual(self.app.session_state.queue_page, 0)
        self.assertTrue(any("No customers match" in m.value for m in self.app.markdown))

    def test_filters_survive_customer_navigation(self):
        self.app.radio(key="queue_scope").set_value("High risk").run()
        self.app.text_input(key="customer_search").set_value("CUST-0004").run()
        self.app.button(key="open-CUST-0004").click().run()
        self.click("Back to review queue")
        self.assertEqual(self.app.radio(key="queue_scope").value, "High risk")
        self.assertEqual(self.app.text_input(key="customer_search").value, "CUST-0004")
        self.assertEqual(len([b for b in self.app.button if b.key and b.key.startswith("open-")]), 1)

    def test_review_removes_from_pending_and_can_reopen(self):
        self.app.button(key="open-CUST-0037").click().run()
        self.click("Mark reviewed")
        self.assertTrue(self.app.session_state.drafts["CUST-0037"]["reviewed"])
        self.assertTrue(self.app.text_area[0].disabled)
        self.assertEqual(len(self.app.get("download_button")), 1)
        self.click("Back to review queue")
        self.assertNotIn("open-CUST-0037", [b.key for b in self.app.button])
        self.app.radio(key="queue_scope").set_value("Reviewed").run()
        self.app.button(key="open-CUST-0037").click().run()
        self.click("Reopen review")
        self.assertFalse(self.app.session_state.drafts["CUST-0037"]["reviewed"])
        self.assertFalse(self.app.text_area[0].disabled)
        self.assertEqual(len(self.app.get("download_button")), 0)

    def test_saved_edits_survive_navigation(self):
        self.app.button(key="open-CUST-0037").click().run()
        message = "Hi, just checking whether you need help with your usual order. Reply whenever it suits you."
        self.app.text_input(key="subject-CUST-0037").set_value("A quick check-in")
        self.app.text_area(key="body-CUST-0037").set_value(message)
        self.click("Save draft")
        self.click("Back to review queue")
        self.app.button(key="open-CUST-0037").click().run()
        self.assertEqual(self.app.text_area[0].value, message)
        self.assertEqual(self.app.text_input(key="subject-CUST-0037").value, "A quick check-in")
        self.assertFalse(self.app.session_state.drafts["CUST-0037"]["reviewed"])

    def test_invalid_edits_cannot_be_reviewed_or_exported(self):
        self.app.button(key="open-CUST-0037").click().run()
        for message in ["Act now for 20% off!!!", "   ", "word " * 131]:
            self.app.text_area(key="body-CUST-0037").set_value(message)
            self.click("Mark reviewed")
            self.assertFalse(self.app.session_state.drafts["CUST-0037"]["reviewed"])
            self.assertEqual(len(self.app.get("download_button")), 0)
            self.assertTrue(self.app.warning)
        self.app.text_input(key="subject-CUST-0037").set_value("   ")
        self.app.text_area(key="body-CUST-0037").set_value("A helpful check-in.")
        self.click("Mark reviewed")
        self.assertFalse(self.app.session_state.drafts["CUST-0037"]["reviewed"])

    def test_on_hold_examples_have_no_response_editor(self):
        for cid in ["CUST-0002", "CUST-0004", "CUST-0005"]:
            self.open_example(cid)
            self.assertEqual(self.app.title[0].value, cid)
            self.assertEqual(len(self.app.text_area), 0)
            self.assertNotIn("Mark reviewed", [b.label for b in self.app.button])
            self.assertTrue(any(x.value == "No outreach for now" for x in self.app.subheader))

    def test_all_customers_and_low_risk_filter(self):
        self.app.button(key="nav-All customers").click().run()
        self.assertTrue(any("200 customers" in c.value for c in self.app.caption))
        self.app.selectbox(key="risk_filter").select("Low").run()
        self.assertFalse(self.app.exception)
        buttons = [b for b in self.app.button if b.key and b.key.startswith("open-")]
        self.assertTrue(buttons)
        self.assertTrue(all(b.label == "View reason" for b in buttons))


if __name__ == "__main__":
    unittest.main()
