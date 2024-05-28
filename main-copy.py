import torch
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
import os
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from helper_functions import load_model, predict_image_efficient_net
from tkinterdnd2 import DND_FILES

from settings import *
from predict_functions import *
import linear_train
import binary_train
import multi_class_train

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class App(ctk.CTk):
    def __init__(self, title, size):
        super().__init__()
        self.title(title)
        self.window_placement(size)

        self.create_widgets()

        self.mainloop()

    def window_placement(self, size):
        display_width = self.winfo_screenwidth()
        display_length = self.winfo_screenheight()

        self.window_width = size[0]
        self.window_height = size[1]

        left = 0.5 * (display_width - self.window_width)
        top = 0.5 * (display_length - self.window_height)

        self._set_appearance_mode('dark')
        self.geometry(f'{self.window_width}x{self.window_height}+{int(left)}+{int(top - 50)}')
        self.minsize(size[0], size[1])
        self.maxsize(size[0], size[1])

    def create_widgets(self):
        self.notebook = ctk.CTkTabview(self, fg_color=BG_COLOR, segmented_button_fg_color=BG_COLOR,
                                       segmented_button_selected_color=BG_COLOR,
                                       segmented_button_unselected_hover_color=BG_COLOR,
                                       segmented_button_selected_hover_color=BG_COLOR, text_color=BG_COLOR,
                                       segmented_button_unselected_color=BG_COLOR)
        self.notebook.place(x=0, y=-15, relh=1, relw=1)

        #TABS
        self.home_tab = self.notebook.add('Home')
        self.train_tab = self.notebook.add('Train')
        self.test_tab = self.notebook.add('Test')

        self.home_tab_widgets()

    def home_tab_widgets(self):
        self.notebook.set('Home')
        self.home_pic(self.home_tab)

        # page intro
        self.page_intro_2 = ctk.CTkLabel(self.home_tab, text='Algorithm at a Time.', font=('Newsreader', 38, 'normal'),
                                         bg_color='transparent')
        self.page_intro_2.place(x=221, y=455)

        self.page_intro_1 = ctk.CTkLabel(self.home_tab, text='Unlocking Potential, One',
                                         font=('Newsreader', 38, 'normal'),
                                         bg_color='transparent')
        self.page_intro_1.place(x=186, y=407)

        # get started button
        font_button = ctk.CTkFont(family='Inter', size=12, weight='bold', underline=True)
        self.start_button = ctk.CTkButton(self.home_tab, width=89, height=29, text='GET STARTED', font=font_button,
                                          command=self.open_ppt, fg_color=BG_COLOR, hover_color=HOVER_COLOR)

        self.start_button.place(x=337, y=516)

        # train button
        self.train_button = ctk.CTkButton(self.home_tab, width=134, height=38, text='Train', fg_color=PRIMARY_COLOR,
                                          border_width=1, border_color=SECONDARY_COLOR,
                                          command=self.open_train_tab)
        self.train_button.place(x=93, y=681)

        self.predict_button = ctk.CTkButton(self.home_tab, width=134, height=38, text='Test', fg_color=PRIMARY_COLOR,
                                            border_width=2, border_color=SECONDARY_COLOR, command=self.open_test_tab)
        self.predict_button.place(x=514, y=681)

    def home_pic(self, master):
        image = Image.open(HOME_PIC_PATH).resize((760, 278))
        self.home_img = ImageTk.PhotoImage(image)

        self.canvas = ctk.CTkCanvas(master, background='#F9F3ED', width=766, height=278, highlightthickness=1,
                                    relief='groove', highlightbackground=SECONDARY_COLOR)
        self.canvas.place(x=-4, y=67)
        self.canvas.bind('<Configure>', self.resize_home_img)

    def resize_home_img(self, event):
        print(event)
        self.canvas.create_image(event.width / 2, event.height / 2, image=self.home_img)

    def open_ppt(self):
        print('Opening PPT FIle')
        os.startfile(PPT_PATH)

    def open_train_tab(self):
        self.notebook.set('Train')
        self.train_tab_widgets()

    def open_test_tab(self):
        self.notebook.set('Test')
        self.test_tab_widgets()

    def open_home_tab(self):
        self.notebook.set('Home')

    def train_tab_widgets(self):
        self.return_button_1 = ctk.CTkButton(self.train_tab, width=205, height=40, text='RETURN',
                                             font=('Newsreader', 19, 'normal'), command=self.open_home_tab,
                                             fg_color=PRIMARY_COLOR)
        self.return_button_1.place(x=292, y=664)

        self.train_label = ctk.CTkLabel(self.train_tab, width=437, height=45, text='Interactive Trainable Models',
                                        font=('Newsreader', 38, 'normal'))
        self.train_label.place(x=57, y=65)

        self.separator = ctk.CTkFrame(self.train_tab, width=630, height=3, bg_color='#dce4ee')
        self.separator.place(x=57, y=140)

        self.combo_var = ctk.StringVar(value='MODEL')
        self.train_dropdown = ctk.CTkComboBox(self.train_tab, variable=self.combo_var, state='readonly', width=105,
                                              height=22,
                                              values=['LINEAR', 'BINARY', 'MULTI-CLASS'], command=self.combobox_event,
                                              fg_color=BG_COLOR, button_hover_color='#144870',
                                              dropdown_hover_color='#144870', border_color=PRIMARY_COLOR,
                                              button_color=PRIMARY_COLOR)
        self.train_dropdown.place(x=64, y=185)

        self.epoch_entry = ctk.CTkEntry(self.train_tab, fg_color='transparent', width=105, height=22,
                                        placeholder_text='EPOCHS #', border_width=3, border_color=PRIMARY_COLOR,
                                        text_color=TEXT_COLOR, placeholder_text_color=TEXT_COLOR)

        self.epoch_entry.place(x=185, y=185)

        self.scrolled_text = ScrolledText(self.train_tab)
        self.scrolled_text.place(x=64, y=226)

        self.graph_frame = ctk.CTkFrame(self.train_tab, width=369, height=300, border_width=2,
                                        border_color=PRIMARY_COLOR,
                                        fg_color=BG_COLOR)
        self.graph_frame.place(x=334, y=192)

        self.create_train_figure()

        self.train_start = ctk.CTkButton(self.train_tab, width=205, height=40, text='START',
                                         font=('Newsreader', 19, 'normal'), command=self.start_model,
                                         fg_color=PRIMARY_COLOR)

        self.train_start.place(x=416, y=516)

    def combobox_event(self, choice):
        self.train_name = self.train_dropdown.get()
        print(self.train_name)
        match self.train_name:
            case 'LINEAR':
                self.scrolled_text.delete('1.0', 'end')
                self.X_train, self.y_train, self.X_test, self.y_test = linear_train.get_data(self.ax_train,
                                                                                             self.train_canvas)
                self.scrolled_text.insert('end', 'Ready...\n\n')
            case 'BINARY':
                self.scrolled_text.delete('1.0', 'end')
                self.X_train, self.y_train, self.X_test, self.y_test = binary_train.get_data(self.ax_train,
                                                                                             self.train_canvas)
                self.scrolled_text.insert('end', 'Ready...\n\n')
            case 'MULTI-CLASS':
                self.scrolled_text.delete('1.0', 'end')
                self.X_train, self.y_train, self.X_test, self.y_test = multi_class_train.get_data(self.ax_train,
                                                                                                  self.train_canvas)
                self.scrolled_text.insert('end', 'Ready...\n\n')

    def create_train_figure(self):
        self.fig_train = Figure(figsize=(3.7, 3), dpi=100, facecolor=BG_COLOR)
        self.ax_train = self.fig_train.add_subplot(111)
        self.ax_train.set_facecolor(BG_COLOR)
        self.ax_train.tick_params(colors=TEXT_COLOR, axis='both')

        self.ax_train.spines['bottom'].set_color(TEXT_COLOR)
        self.ax_train.spines['top'].set_color(TEXT_COLOR)
        self.ax_train.spines['right'].set_color(TEXT_COLOR)
        self.ax_train.spines['left'].set_color(TEXT_COLOR)

        self.train_canvas = FigureCanvasTkAgg(self.fig_train, master=self.graph_frame)
        self.train_canvas.get_tk_widget().pack(expand=True, padx=5, pady=5)
        self.ax_train.clear()
        self.train_canvas.draw()

    def test_tab_widgets(self):
        self.return_button_2 = ctk.CTkButton(self.test_tab, width=205, height=40, text='RETURN',
                                             font=('Newsreader', 19, 'normal'), command=self.open_home_tab,
                                             fg_color=PRIMARY_COLOR)
        self.return_button_2.place(x=269, y=664)

        self.train_label = ctk.CTkLabel(self.test_tab, width=284, height=45, text='Pretrained Models',
                                        font=('Newsreader', 38, 'normal'))
        self.train_label.place(x=57, y=65)

        self.separator = ctk.CTkFrame(self.test_tab, width=630, height=3, bg_color='#dce4ee')
        self.separator.place(x=57, y=140)

        self.predict_frame = ctk.CTkFrame(self.test_tab, width=406, height=270, border_width=2,
                                          border_color=PRIMARY_COLOR,
                                          fg_color=BG_COLOR)
        self.predict_frame.place(x=168, y=244)

        self.create_predict_figure()

        self.model_var = ctk.StringVar(value='MODEL')
        self.train_dropdown = ctk.CTkComboBox(self.test_tab, variable=self.model_var, state='readonly', width=155,
                                              height=22,
                                              values=['MNIST', 'CATS VS DOGS', 'WASTE CLASS'], command=self.model_event,
                                              fg_color=BG_COLOR, button_hover_color='#144870',
                                              dropdown_hover_color='#144870', border_color=PRIMARY_COLOR,
                                              button_color=PRIMARY_COLOR)
        self.train_dropdown.place(x=310, y=208)

        self.status_label = ctk.CTkLabel(self.test_tab, width=109, height=38, text=None,
                                         font=('Newsreader', 24, 'normal'))
        self.status_label.place(x=370, y=550, anchor='center')

    def start_model(self):
        self.epochs = self.epoch_entry.get()
        self.scrolled_text.insert('end', '\nProcessing...\n\n')
        if self.epochs.isdigit():
            match self.train_name:
                case 'LINEAR':
                    self.model, self.loss_fn, self.optimizer = linear_train.start_training()
                    train_model(model=self.model,
                                train=(self.X_train, self.y_train),
                                test=(self.X_test, self.y_test),
                                loss_fn=self.loss_fn,
                                optimizer=self.optimizer,
                                epochs=self.epochs,
                                widget=self.scrolled_text)

                case 'BINARY':
                    print(len(self.X_train))
                    self.model, self.loss_fn, self.optimizer = binary_train.start_training()
                    train_test_step_binary(model=self.model,
                                           x_train=self.X_train,
                                           y_train=self.y_train,
                                           x_test=self.X_test,
                                           y_test=self.y_test,
                                           loss_fn=self.loss_fn,
                                           optimizer=self.optimizer,
                                           epochs=self.epochs,
                                           widget=self.scrolled_text)

                case 'MULTI-CLASS':
                    self.model, self.loss_fn, self.optimizer = multi_class_train.start_training()
                    train_test_step_multi(model=self.model,
                                          x_train=self.X_train,
                                          y_train=self.y_train,
                                          x_test=self.X_test,
                                          y_test=self.y_test,
                                          loss_fn=self.loss_fn,
                                          optimizer=self.optimizer,
                                          epochs=self.epochs,
                                          widget=self.scrolled_text)

            with torch.inference_mode():
                test_pred = self.model(self.X_test)

            match self.train_name:
                case 'LINEAR':
                    plot_predictions_linear(train_data=self.X_train, train_labels=self.y_train, test_data=self.X_test,
                                            test_labels=self.y_test, predictions=test_pred, ax=self.ax_train,
                                            canvas=self.train_canvas)
                case _ if 'BINARY' or 'MULTI-CLASS':
                    plot_decision_boundary(model=self.model, X=self.X_test, y=self.y_test, ax=self.ax_train,
                                           canvas=self.train_canvas)


    def model_event(self, choice):
        self.chosen_predictor = self.train_dropdown.get()
        self.predict_frame.drop_target_register(DND_FILES)
        self.predict_frame.dnd_bind('<<Drop>>', self.start_predict)
        match self.chosen_predictor:
            case 'MNIST':
                self.status_label.configure(text='Loading...')
                self.mnist_model_0 = load_model(model='mnist', path=PATH_MNIST).to(device)
                self.status_label.configure(text='Ready')

            case 'CATS VS DOGS':
                self.status_label.configure(text='Loading...')
                self.cats_dogs_model_3 = load_model(model='CVD', path=PATH_CVD).to(device)
                self.status_label.configure(text='Ready')

            case 'WASTE CLASS':
                self.status_label.configure(text='Loading...')

    def create_predict_figure(self):
        self.fig = Figure(figsize=(4.1, 2.7), dpi=100, facecolor=BG_COLOR)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG_COLOR)
        self.ax.tick_params(colors=TEXT_COLOR, axis='both')

        self.ax.spines['bottom'].set_color(TEXT_COLOR)
        self.ax.spines['top'].set_color(TEXT_COLOR)
        self.ax.spines['right'].set_color(TEXT_COLOR)
        self.ax.spines['left'].set_color(TEXT_COLOR)

        self.predict_canvas = FigureCanvasTkAgg(self.fig, master=self.predict_frame)
        self.predict_canvas.get_tk_widget().pack(expand=True, padx=5, pady=5)
        self.ax.clear()
        self.predict_canvas.draw()

    def start_predict(self, event):
        self.ax.clear()
        self.predict_canvas.draw()

        IMG_PATH = event.data
        IMG_PATH = IMG_PATH.replace("{", "").replace("}", "")
        IMG_PATH = IMG_PATH.replace("\\", "/")

        self.status_label.configure(text='Processing...')
        match self.chosen_predictor:
            case 'MNIST':
                img, pred, probs = predict_image_mnist(model=self.mnist_model_0,
                                                       image_path=IMG_PATH,
                                                       device=device)
            case 'CATS VS DOGS':
                img, pred, probs = predict_image_efficient_net(model=self.cats_dogs_model_3, image_path=IMG_PATH,
                                                               device=device)
            case 'WASTE CLASS':
                pass

        self.status_label.configure(text='Done')

        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title(f"Pred: {pred} | Prob: {probs:.2f}%").set_color(TEXT_COLOR)
        self.ax.axis(True)

        self.predict_canvas.draw()


class ScrolledText(ctk.CTkTextbox):
    def __init__(self, parent):
        super().__init__(master=parent, width=229, height=330, border_color=PRIMARY_COLOR, border_width=2,
                         fg_color=BG_COLOR, state='normal')


def train_step(model: torch.nn.Module,
               x,
               y,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    # 1. Forward pass
    y_pred = model(x)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()
    return loss


def test_step(model: torch.nn.Module,
              x,
              y,
              loss_fn: torch.nn.Module, ):
    model.eval()  # put model_codes in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model(x)

        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_pred, y)

    return test_loss


def plot_predictions_linear(train_data, train_labels, test_data, test_labels, predictions=None, ax=None, canvas=None):
    ax.clear()
    ax.scatter(train_data, train_labels, c='b', s=4, label='Training Data')
    ax.scatter(test_data, test_labels, c='g', s=4, label='Testing Data')

    if predictions is not None:
        ax.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    ax.legend(prop={'size': 5})
    canvas.draw()


def train_model(model: torch.nn.Module, train, test, loss_fn, optimizer, epochs, widget):
    x_train, y_train = train[0], train[1]
    x_test, y_test = test[0], test[1]

    for epoch in range(int(epochs)):
        train_loss = train_step(model=model, x=x_train, y=y_train, loss_fn=loss_fn, optimizer=optimizer)
        test_loss = test_step(model=model, x=x_test, y=y_test, loss_fn=loss_fn)
        if epoch % 100 == 0:
            widget.insert('end', f'\nEpoch {epoch + 1} | Train Loss: {train_loss} | Test Loss: {test_loss}\n')


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, ax, canvas):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    ax.clear()
    ax.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    canvas.draw()


def train_test_step_binary(model: torch.nn.Module,
                           x_train: torch.float32, y_train: torch.float32,
                           x_test: torch.float32, y_test: torch.float32,
                           loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, widget):
    model.train()

    for epoch in range(int(epochs)):
        y_logits = model(x_train).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test).squeeze()
            test_preds = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits, y_test)

        if epoch % 100 == 0:
            widget.insert('end', f'\nEpoch {epoch} | Train Loss: {loss:.5f} | Test Loss: {test_loss:.5f}\n')
            print(f'\nEpoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f}')


def train_test_step_multi(model: torch.nn.Module,
                          x_train: torch.float32, y_train: torch.float32,
                          x_test: torch.float32, y_test: torch.float32,
                          loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, widget):
    model.train()

    for epoch in range(int(epochs)):
        y_logits = model(x_train)
        y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

        loss = loss_fn(y_logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test).squeeze()
            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

            test_loss = loss_fn(test_logits, y_test)

        if epoch % 100 == 0:
            widget.insert('end', f'\nEpoch {epoch} | Train Loss: {loss:.5f} | Test Loss: {test_loss:.5f}\n')
            print(f'\nEpoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f}%')


if __name__ == '__main__':
    App('Neuron', size=(760, 772))
