from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Stock
from django.contrib.auth.models import User


class CustomUserCreationForm(UserCreationForm):
    preferred_stocks = forms.ModelMultipleChoiceField(
        queryset=Stock.objects.all(),
        widget=forms.CheckboxSelectMultiple(),
        required=False,
        help_text="Select the stocks you want to see in your dashboard"
    )

    class Meta:
        model = User
        fields = UserCreationForm.Meta.fields + ('preferred_stocks',)

class StockSelectionForm(forms.Form):
    selected_stock = forms.ModelChoiceField(
        queryset=Stock.objects.all(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        required=True,
        label="Select a stock to predict"
    )
