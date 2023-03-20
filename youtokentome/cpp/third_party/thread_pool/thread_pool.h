// Copyright (c) 2023 Ibragim Dzhiblavi
// Modified 2023 Gleb Koveshnikov

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vkcom {

class ThreadPool {
  public:
    using Task = std::function<void()>;

  public:
    ThreadPool(size_t thread_count) {
        if (thread_count == 0) {
            thread_count = static_cast<size_t>(std::thread::hardware_concurrency());
        }
        if (thread_count == 0) {
            thread_count = 8;
        }
        for (size_t thread = 0; thread < thread_count; ++thread) {
            threads_.emplace_back([this] {
                while (!stop_.load(std::memory_order_relaxed)) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    work_cv_.wait(lock, [this] {
                        return stop_.load(std::memory_order_relaxed) || !task_queue_.empty();
                    });
                    if (stop_.load(std::memory_order_relaxed)) {
                        break;
                    }
                    if (task_queue_.empty()) {
                        continue;
                    }
                    ++active_tasks_;
                    auto task = std::move(task_queue_.front());
                    task_queue_.pop();
                    lock.unlock();
                    task();
                    lock.lock();
                    --active_tasks_;
                    complete_cv_.notify_one();
                }
            });
        }
    }

    ~ThreadPool() {
        stop_.store(true, std::memory_order_relaxed);
        work_cv_.notify_all();
        for (auto &thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void submit(Task &&task) {
        {
            std::lock_guard<std::mutex> lg(mutex_);
            task_queue_.emplace(std::move(task));
        }
        work_cv_.notify_one();
    }

    void waitCompletion() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (active_tasks_ != 0 || !task_queue_.empty()) {
            complete_cv_.wait(lock, [this] { return active_tasks_ == 0 && task_queue_.empty(); });
        }
    }

    [[nodiscard]] size_t maxThreads() const noexcept { return threads_.size(); }

  private:
    std::atomic<bool> stop_{false};
    size_t active_tasks_{0};
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable complete_cv_;
    std::vector<std::thread> threads_;
    std::queue<Task> task_queue_;
};

} // namespace vkcom