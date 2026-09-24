import { test, expect } from "@playwright/test";

test("desktop studio, guides, action library and source controls", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Hiểu từng chuyển động." }),
  ).toBeVisible();
  await expect(page.getByText("Mô hình sẵn sàng")).toBeVisible();
  await page.screenshot({
    path: "test-results/studio-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Hướng dẫn demo" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button", { name: "Đã hiểu, bắt đầu thôi" }).click();
  await page
    .getByRole("button", { name: "Gác chân Hành động cần chú ý" })
    .click();
  await expect(
    page.getByRole("dialog").getByRole("heading", { name: "Gác chân" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Đóng", exact: true }).click();
  await page.getByRole("button", { name: "Video tải lên" }).click();
  await expect(
    page.getByText("Chọn video MP4 hoặc WebM để bắt đầu"),
  ).toBeVisible();
  await page.getByRole("switch", { name: "Hiển thị khung xương" }).click();
  await expect(page.getByRole("switch")).toHaveAttribute(
    "aria-checked",
    "false",
  );
  expect(errors).toEqual([]);
});

test("mobile layout has no horizontal overflow and retains primary actions", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Bắt đầu nhận diện" }),
  ).toBeVisible();
  const overflow = await page.locator("body *").evaluateAll((elements) =>
    elements
      .filter(
        (element) =>
          element.getBoundingClientRect().right > window.innerWidth + 1,
      )
      .map((element) => ({
        tag: element.tagName,
        class: element.className,
        right: element.getBoundingClientRect().right,
      }))
      .slice(0, 20),
  );
  if (overflow.length) console.log("Overflow:", JSON.stringify(overflow));
  await page.screenshot({
    path: "test-results/mobile-debug.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/studio-mobile.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Thư viện hành động", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Một mô hình. Sáu hành động." }),
  ).toBeInViewport();
});

test("unavailable backend produces a visible recoverable error", async ({
  page,
}) => {
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: {
        status: "unavailable",
        message: "Không nạp được model kiểm thử.",
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Bắt đầu nhận diện" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Không nạp được model kiểm thử.",
  );
  await expect(
    page.getByRole("button", { name: "Bắt đầu nhận diện" }),
  ).toBeEnabled();
});

test("camera denial does not leave a running session", async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      throw new DOMException("denied", "NotAllowedError");
    };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Bắt đầu nhận diện" }).click();
  await expect(page.getByRole("alert")).toContainText("Chưa có quyền");
  await expect(
    page.getByRole("button", { name: "Bắt đầu nhận diện" }),
  ).toBeEnabled();
});

test("wide camera keeps the skeleton canvas aligned on mobile and expanded view", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = 960;
      canvas.height = 540;
      const ctx = canvas.getContext("2d")!;
      const stream = canvas.captureStream(10);
      setInterval(() => {
        ctx.fillStyle = "#eee";
        ctx.fillRect(0, 0, 960, 540);
      }, 100);
      return stream;
    };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Bắt đầu nhận diện" }).click();
  await expect(page.locator("video.visible")).toBeVisible();
  const stage = page.locator(".camera-stage");
  await expect
    .poll(async () => {
      const box = await stage.boundingBox();
      return box!.width / box!.height;
    })
    .toBeCloseTo(16 / 9, 2);
  await page.getByRole("button", { name: "Mở rộng khung hình" }).click();
  await expect
    .poll(async () => {
      const box = await stage.boundingBox();
      return box!.width / box!.height;
    })
    .toBeCloseTo(16 / 9, 2);
  await page.getByRole("button", { name: "Thu nhỏ khung hình" }).click();
  await page.getByRole("button", { name: "Dừng nhận diện" }).click();
});

test("uploaded video receives real model predictions and exports history", async ({
  page,
}) => {
  test.skip(
    !process.env.POSE_TEST_VIDEO,
    "Set POSE_TEST_VIDEO to a local test video with a visible person.",
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Video tải lên" }).click();
  await page
    .getByLabel("Chọn video để nhận diện")
    .setInputFiles(process.env.POSE_TEST_VIDEO!);
  await page.getByRole("button", { name: "Bắt đầu nhận diện" }).click();
  await expect(page.locator(".confidence-chip")).toContainText("% độ tin cậy", {
    timeout: 30000,
  });
  await expect(page.locator("tbody tr").first()).toBeVisible();
  await page.screenshot({
    path: "test-results/studio-live.png",
    fullPage: true,
  });
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Xuất CSV" }).click();
  expect((await download).suggestedFilename()).toBe("pose-studio-history.csv");
  await page.getByRole("button", { name: "Dừng nhận diện" }).click();
  await expect(
    page.getByRole("heading", { name: "Chưa có kết quả" }),
  ).toBeVisible();
  expect(
    await page
      .locator("video")
      .evaluate(
        (video: HTMLVideoElement) =>
          video.srcObject === null && !video.getAttribute("src"),
      ),
  ).toBe(true);
  await expect(page.locator("tbody tr").first()).toBeVisible();
  await page.getByRole("button", { name: "Xóa lịch sử" }).click();
  await expect(page.getByText("Mọi chuyển động bắt đầu từ đây")).toBeVisible();
});
